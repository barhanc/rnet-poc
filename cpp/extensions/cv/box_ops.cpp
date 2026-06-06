#include "box_ops.h"

#include <cmath>
#include <stdexcept>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "core/tensor.h"
#include "core/dtype.h"

namespace mylib::extensions::cv::box_ops
{
    namespace jsi = facebook::jsi;
    using TensorHostObject = mylib::core::tensor::TensorHostObject;

    namespace
    {
        enum class BoxFormat
        {
            XYXY,
            XYWH,
            CXCYWH
        };

        BoxFormat parseBoxFormat(const std::string &s)
        {
            if (s == "xyxy")
            {
                return BoxFormat::XYXY;
            }
            else if (s == "xywh")
            {
                return BoxFormat::XYWH;
            }
            else if (s == "cxcywh")
            {
                return BoxFormat::CXCYWH;
            }
            throw std::invalid_argument("unsupported boxFormat '" + s + "'");
        }

        std::tuple<float, float, float, float> decodeToXyxy(
            float a, float b, float c, float d,
            BoxFormat format
        )
        {
            switch (format)
            {
                case BoxFormat::XYXY:
                    return {a, b, c, d};
                case BoxFormat::XYWH:
                    return {a, b, a + c, b + d};
                case BoxFormat::CXCYWH:
                    return {a - c / 2.0f, b - d / 2.0f, a + c / 2.0f, b + d / 2.0f};
            }
        }
    } // namespace

    void install_nms(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "nms";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count < 3)
            {
                throw jsi::JSError(rt, "Usage: nms(boxes, scores, options)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "nms: boxes must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "nms: scores must be a Tensor");
            }

            if (!args[2].isObject())
            {
                throw jsi::JSError(rt, "nms: options must be an object");
            }

            auto boxes = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto scores = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            if (boxes.get() == scores.get())
            {
                throw jsi::JSError(rt, "nms: boxes and scores cannot be the same tensor.");
            }

            auto opts = args[2].asObject(rt);

            if (!opts.hasProperty(rt, "iouThreshold"))
            {
                throw jsi::JSError(rt, "nms: options.iouThreshold is required");
            }
            auto iouVal = opts.getProperty(rt, "iouThreshold");
            if (!iouVal.isNumber())
            {
                throw jsi::JSError(rt, "nms: options.iouThreshold must be a number");
            }
            double iouThreshold = iouVal.asNumber();

            if (!opts.hasProperty(rt, "scoreThreshold"))
            {
                throw jsi::JSError(rt, "nms: options.scoreThreshold is required");
            }
            auto scoreVal = opts.getProperty(rt, "scoreThreshold");
            if (!scoreVal.isNumber())
            {
                throw jsi::JSError(rt, "nms: options.scoreThreshold must be a number");
            }
            double scoreThreshold = scoreVal.asNumber();

            if (!opts.hasProperty(rt, "boxFormat"))
            {
                throw jsi::JSError(rt, "nms: options.boxFormat is required");
            }
            auto formatVal = opts.getProperty(rt, "boxFormat");
            if (!formatVal.isString())
            {
                throw jsi::JSError(rt, "nms: options.boxFormat must be a string");
            }
            std::string boxFormatStr = formatVal.asString(rt).utf8(rt);

            BoxFormat boxFormat;
            try
            {
                boxFormat = parseBoxFormat(boxFormatStr);
            }
            catch (const std::invalid_argument &e)
            {
                throw jsi::JSError(rt, std::string("nms: ") + e.what());
            }

            if (boxes->shape_.size() != 2 || boxes->shape_[1] != 4)
            {
                throw jsi::JSError(rt, "nms: boxes must be a 2D tensor with shape [N, 4]");
            }

            if (scores->shape_.size() != 1)
            {
                throw jsi::JSError(rt, "nms: scores must be a 1D tensor with shape [N]");
            }

            int N = boxes->shape_[0];
            if (scores->shape_[0] != N)
            {
                throw jsi::JSError(rt, "nms: boxes and scores must have the same number of elements (N)");
            }

            if (boxes->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "nms: boxes must have dtype float32");
            }

            if (scores->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "nms: scores must have dtype float32");
            }

            // Acquire shared locks (read-only on both tensors)
            std::shared_lock<std::shared_mutex> boxes_lock(boxes->mutex_, std::try_to_lock);
            if (!boxes_lock.owns_lock())
            {
                throw jsi::JSError(rt, "nms: boxes tensor is currently in use");
            }

            std::shared_lock<std::shared_mutex> scores_lock(scores->mutex_, std::try_to_lock);
            if (!scores_lock.owns_lock())
            {
                throw jsi::JSError(rt, "nms: scores tensor is currently in use");
            }

            if (!boxes->data_)
            {
                throw jsi::JSError(rt, "nms: boxes tensor has been disposed");
            }

            if (!scores->data_)
            {
                throw jsi::JSError(rt, "nms: scores tensor has been disposed");
            }

            const float *boxes_ptr = reinterpret_cast<const float *>(boxes->data_.get());
            const float *scores_ptr = reinterpret_cast<const float *>(scores->data_.get());

            std::vector<std::pair<int, float>> candidates;
            candidates.reserve(N);
            for (int i = 0; i < N; ++i)
            {
                if (scores_ptr[i] > scoreThreshold)
                {
                    candidates.push_back({i, scores_ptr[i]});
                }
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                      { return a.second > b.second; });

            std::vector<int> kept;
            std::vector<bool> suppressed(candidates.size(), false);

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (suppressed[i])
                    continue;

                int idx_i = candidates[i].first;
                kept.push_back(idx_i);

                float x1_a = 0, y1_a = 0, x2_a = 0, y2_a = 0;
                std::tie(x1_a, y1_a, x2_a, y2_a) = decodeToXyxy(
                    boxes_ptr[idx_i * 4 + 0],
                    boxes_ptr[idx_i * 4 + 1],
                    boxes_ptr[idx_i * 4 + 2],
                    boxes_ptr[idx_i * 4 + 3],
                    boxFormat
                );
                float area_a = (x2_a - x1_a) * (y2_a - y1_a);

                for (size_t j = i + 1; j < candidates.size(); ++j)
                {
                    if (suppressed[j])
                        continue;

                    int idx_j = candidates[j].first;
                    float x1_b = 0, y1_b = 0, x2_b = 0, y2_b = 0;
                    std::tie(x1_b, y1_b, x2_b, y2_b) = decodeToXyxy(
                        boxes_ptr[idx_j * 4 + 0],
                        boxes_ptr[idx_j * 4 + 1],
                        boxes_ptr[idx_j * 4 + 2],
                        boxes_ptr[idx_j * 4 + 3],
                        boxFormat
                    );
                    float area_b = (x2_b - x1_b) * (y2_b - y1_b);

                    float inter_x1 = std::max(x1_a, x1_b);
                    float inter_y1 = std::max(y1_a, y1_b);
                    float inter_x2 = std::min(x2_a, x2_b);
                    float inter_y2 = std::min(y2_a, y2_b);

                    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
                    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
                    float intersection = inter_w * inter_h;

                    float union_area = area_a + area_b - intersection;
                    float iou = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;

                    if (iou > static_cast<float>(iouThreshold))
                    {
                        suppressed[j] = true;
                    }
                }
            }

            jsi::Array result = jsi::Array(rt, kept.size());
            for (size_t i = 0; i < kept.size(); ++i)
            {
                result.setValueAtIndex(rt, i, jsi::Value(kept[i]));
            }

            return result;
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }

    void install_weightedNms(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "weightedNms";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count < 3)
            {
                throw jsi::JSError(rt, "Usage: weightedNms(boxes, scores, options)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt) ||
                !args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "weightedNms: boxes and scores must be Tensors");
            }

            if (!args[2].isObject())
            {
                throw jsi::JSError(rt, "weightedNms: options must be an object");
            }

            auto boxes = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto scores = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto opts = args[2].asObject(rt);

            if (!opts.hasProperty(rt, "suppressionThreshold") || !opts.hasProperty(rt, "boxFormat") || !opts.hasProperty(rt, "scoreThreshold"))
            {
                throw jsi::JSError(rt, "weightedNms: options must specify suppressionThreshold, boxFormat, and scoreThreshold");
            }

            float suppressionThreshold = static_cast<float>(opts.getProperty(rt, "suppressionThreshold").asNumber());
            float scoreThreshold = static_cast<float>(opts.getProperty(rt, "scoreThreshold").asNumber());
            std::string boxFormatStr = opts.getProperty(rt, "boxFormat").asString(rt).utf8(rt);
            BoxFormat format = parseBoxFormat(boxFormatStr);

            // Locks
            std::shared_lock<std::shared_mutex> boxes_lock(boxes->mutex_, std::try_to_lock);
            std::shared_lock<std::shared_mutex> scores_lock(scores->mutex_, std::try_to_lock);

            if (!boxes_lock.owns_lock() || !scores_lock.owns_lock())
            {
                throw jsi::JSError(rt, "weightedNms: one of the tensors is currently locked");
            }

            if (!boxes->data_ || !scores->data_)
            {
                throw jsi::JSError(rt, "weightedNms: tensors must not be disposed");
            }

            if (scores->shape_.size() != 3 || scores->shape_[0] != 1 || scores->shape_[2] != 1)
            {
                throw jsi::JSError(rt, "weightedNms: scores must be a 3D tensor with shape [1, N, 1]");
            }
            int N = scores->shape_[1];

            if (boxes->shape_.size() != 3 || boxes->shape_[0] != 1 || boxes->shape_[2] != 16)
            {
                throw jsi::JSError(rt, "weightedNms: boxes must be a 3D tensor with shape [1, N, 16]");
            }

            const float *boxes_ptr = reinterpret_cast<const float *>(boxes->data_.get());
            const float *scores_ptr = reinterpret_cast<const float *>(scores->data_.get());

            struct Candidate {
                int index;
                float score;
                float ymin, xmin, ymax, xmax;
            };

            std::vector<Candidate> candidates;
            candidates.reserve(N);

            for (int idx = 0; idx < N; ++idx)
            {
                float raw_score = scores_ptr[idx];
                float score = 1.0f / (1.0f + std::exp(-raw_score));

                if (score >= scoreThreshold)
                {
                    float a = boxes_ptr[idx * 16 + 0];
                    float b = boxes_ptr[idx * 16 + 1];
                    float c = boxes_ptr[idx * 16 + 2];
                    float d = boxes_ptr[idx * 16 + 3];

                    auto [xmin, ymin, xmax, ymax] = decodeToXyxy(a, b, c, d, format);
                    candidates.push_back({idx, score, ymin, xmin, ymax, xmax});
                }
            }

            if (candidates.empty())
            {
                return jsi::Array(rt, 0);
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate &a, const Candidate &b) {
                          return a.score > b.score;
                      });

            std::vector<std::vector<int>> groups;
            std::vector<bool> suppressed(candidates.size(), false);

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (suppressed[i])
                    continue;

                const auto &cand_a = candidates[i];
                float area_a = (cand_a.ymax - cand_a.ymin) * (cand_a.xmax - cand_a.xmin);

                std::vector<int> overlapping = { cand_a.index };

                for (size_t j = i + 1; j < candidates.size(); ++j)
                {
                    if (suppressed[j])
                        continue;

                    const auto &cand_b = candidates[j];
                    float area_b = (cand_b.ymax - cand_b.ymin) * (cand_b.xmax - cand_b.xmin);

                    float inter_ymin = std::max(cand_a.ymin, cand_b.ymin);
                    float inter_xmin = std::max(cand_a.xmin, cand_b.xmin);
                    float inter_ymax = std::min(cand_a.ymax, cand_b.ymax);
                    float inter_xmax = std::min(cand_a.xmax, cand_b.xmax);

                    float inter_h = std::max(0.0f, inter_ymax - inter_ymin);
                    float inter_w = std::max(0.0f, inter_xmax - inter_xmin);
                    float intersection = inter_h * inter_w;

                    float union_area = area_a + area_b - intersection;
                    float iou = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;

                    if (iou > suppressionThreshold)
                    {
                        overlapping.push_back(cand_b.index);
                        suppressed[j] = true;
                    }
                }

                groups.push_back(std::move(overlapping));
            }

            jsi::Array resultGroups = jsi::Array(rt, groups.size());
            for (size_t i = 0; i < groups.size(); ++i)
            {
                jsi::Array singleGroup = jsi::Array(rt, groups[i].size());
                for (size_t j = 0; j < groups[i].size(); ++j)
                {
                    singleGroup.setValueAtIndex(rt, j, jsi::Value(groups[i][j]));
                }
                resultGroups.setValueAtIndex(rt, i, singleGroup);
            }

            return resultGroups;
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 5, fnBody));
    }
} // namespace mylib::extensions::cv::box_ops
