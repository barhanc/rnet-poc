#include "box_ops.h"

#include <array>
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

        enum class NmsType
        {
            Standard,
            Weighted
        };

        NmsType parseNmsType(const std::string &s)
        {
            if (s == "standard")
            {
                return NmsType::Standard;
            }
            else if (s == "weighted")
            {
                return NmsType::Weighted;
            }
            throw std::invalid_argument("unsupported nmsType '" + s + "'");
        }

        std::array<float, 4> decodeToXyxy(
            float a, float b, float c, float d,
            BoxFormat format)
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

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt) ||
                !args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "nms: boxes and scores must be Tensors");
            }

            if (!args[2].isObject())
            {
                throw jsi::JSError(rt, "nms: options must be an object");
            }

            auto boxes = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto scores = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto opts = args[2].asObject(rt);

            if (boxes.get() == scores.get())
            {
                throw jsi::JSError(rt, "nms: boxes and scores cannot be the same tensor.");
            }

            if (!opts.hasProperty(rt, "iouThreshold") ||
                !opts.hasProperty(rt, "boxFormat") ||
                !opts.hasProperty(rt, "confidenceThreshold") ||
                !opts.hasProperty(rt, "nmsType"))
            {
                throw jsi::JSError(rt, "nms: options must specify iouThreshold, boxFormat, confidenceThreshold, and nmsType");
            }

            float iouThreshold = static_cast<float>(opts.getProperty(rt, "iouThreshold").asNumber());
            float confidenceThreshold = static_cast<float>(opts.getProperty(rt, "confidenceThreshold").asNumber());

            std::string nmsTypeStr = opts.getProperty(rt, "nmsType").asString(rt).utf8(rt);
            std::string boxFormatStr = opts.getProperty(rt, "boxFormat").asString(rt).utf8(rt);

            NmsType nmsType;
            BoxFormat boxFormat;
            try
            {
                nmsType = parseNmsType(nmsTypeStr);
                boxFormat = parseBoxFormat(boxFormatStr);
            }
            catch (const std::invalid_argument &e)
            {
                throw jsi::JSError(rt, "nms: " + std::string(e.what()));
            }

            std::shared_lock<std::shared_mutex> boxes_lock(boxes->mutex_, std::try_to_lock);
            std::shared_lock<std::shared_mutex> scores_lock(scores->mutex_, std::try_to_lock);

            if (!boxes_lock.owns_lock() || !scores_lock.owns_lock())
            {
                throw jsi::JSError(rt, "nms: one of the tensors is currently locked");
            }

            if (!boxes->data_ || !scores->data_)
            {
                throw jsi::JSError(rt, "nms: tensors must not be disposed");
            }

            if (scores->shape_.size() != 1)
            {
                throw jsi::JSError(rt, "nms: scores must be a 1D tensor with shape [N]");
            }
            std::int32_t numAnchors = scores->shape_[0];

            if (boxes->shape_.size() != 2 || boxes->shape_[1] != 4)
            {
                throw jsi::JSError(rt, "nms: boxes must be a 2D tensor with shape [N, 4]");
            }

            if (boxes->shape_[0] != numAnchors)
            {
                throw jsi::JSError(rt, "nms: boxes and scores must have the same number of elements");
            }

            if (boxes->dtype_ != mylib::core::types::DType::float32 || scores->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "nms: boxes and scores must have dtype float32");
            }

            const float *boxes_ptr = reinterpret_cast<const float *>(boxes->data_.get());
            const float *scores_ptr = reinterpret_cast<const float *>(scores->data_.get());

            std::vector<std::pair<std::int32_t, float>> candidates;
            candidates.reserve(numAnchors);

            for (size_t idx = 0; idx < numAnchors; ++idx)
            {
                float score = scores_ptr[idx];

                if (score >= confidenceThreshold)
                {
                    candidates.push_back({idx, score});
                }
            }

            if (candidates.empty())
            {
                return jsi::Array(rt, 0);
            }

            std::ranges::sort(candidates, [](const auto &lhs, const auto &rhs)
                              { return lhs.second > rhs.second; });

            std::vector<std::vector<std::int32_t>> groups;
            std::vector<bool> suppressed(candidates.size(), false);

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (suppressed[i])
                {
                    continue;
                }

                std::int32_t idx_i = candidates[i].first;

                auto [xmin_a, ymin_a, xmax_a, ymax_a] = decodeToXyxy(
                    boxes_ptr[idx_i * 4 + 0],
                    boxes_ptr[idx_i * 4 + 1],
                    boxes_ptr[idx_i * 4 + 2],
                    boxes_ptr[idx_i * 4 + 3],
                    boxFormat);

                float area_a = (xmax_a - xmin_a) * (ymax_a - ymin_a);

                std::vector<std::int32_t> overlapping = {idx_i};

                for (size_t j = i + 1; j < candidates.size(); ++j)
                {
                    if (suppressed[j])
                    {
                        continue;
                    }

                    std::int32_t idx_j = candidates[j].first;

                    auto [xmin_b, ymin_b, xmax_b, ymax_b] = decodeToXyxy(
                        boxes_ptr[idx_j * 4 + 0],
                        boxes_ptr[idx_j * 4 + 1],
                        boxes_ptr[idx_j * 4 + 2],
                        boxes_ptr[idx_j * 4 + 3],
                        boxFormat);

                    float area_b = (xmax_b - xmin_b) * (ymax_b - ymin_b);

                    float inter_ymin = std::max(ymin_a, ymin_b);
                    float inter_xmin = std::max(xmin_a, xmin_b);
                    float inter_ymax = std::min(ymax_a, ymax_b);
                    float inter_xmax = std::min(xmax_a, xmax_b);

                    float inter_h = std::max(0.0f, inter_ymax - inter_ymin);
                    float inter_w = std::max(0.0f, inter_xmax - inter_xmin);
                    float intersection = inter_h * inter_w;

                    float union_area = area_a + area_b - intersection;
                    float iou = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;

                    if (iou > iouThreshold)
                    {
                        if (nmsType == NmsType::Weighted)
                        {
                            overlapping.push_back(idx_j);
                        }
                        suppressed[j] = true;
                    }
                }

                groups.push_back(std::move(overlapping));
            }

            switch (nmsType)
            {
            case NmsType::Standard:
            {
                jsi::Array result = jsi::Array(rt, groups.size());
                for (size_t i = 0; i < groups.size(); ++i)
                {
                    result.setValueAtIndex(rt, i, jsi::Value(static_cast<double>(groups[i][0])));
                }
                return result;
            }
            case NmsType::Weighted:
            {
                jsi::Array resultGroups = jsi::Array(rt, groups.size());
                for (size_t i = 0; i < groups.size(); ++i)
                {
                    jsi::Array singleGroup = jsi::Array(rt, groups[i].size());
                    for (size_t j = 0; j < groups[i].size(); ++j)
                    {
                        singleGroup.setValueAtIndex(rt, j, jsi::Value(static_cast<double>(groups[i][j])));
                    }
                    resultGroups.setValueAtIndex(rt, i, singleGroup);
                }
                return resultGroups;
            }
            }
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }
} // namespace mylib::extensions::cv::box_ops
