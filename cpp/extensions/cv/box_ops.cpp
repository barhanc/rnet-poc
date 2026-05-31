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

                float x1_a = boxes_ptr[idx_i * 4 + 0];
                float y1_a = boxes_ptr[idx_i * 4 + 1];
                float x2_a = boxes_ptr[idx_i * 4 + 2];
                float y2_a = boxes_ptr[idx_i * 4 + 3];
                float area_a = (x2_a - x1_a) * (y2_a - y1_a);

                for (size_t j = i + 1; j < candidates.size(); ++j)
                {
                    if (suppressed[j])
                        continue;

                    int idx_j = candidates[j].first;
                    float x1_b = boxes_ptr[idx_j * 4 + 0];
                    float y1_b = boxes_ptr[idx_j * 4 + 1];
                    float x2_b = boxes_ptr[idx_j * 4 + 2];
                    float y2_b = boxes_ptr[idx_j * 4 + 3];
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

    void install_decodeBoxes(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "decodeBoxes";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 3)
            {
                throw jsi::JSError(rt, "Usage: decodeBoxes(src, dst, options)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "decodeBoxes: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "decodeBoxes: dst must be a Tensor");
            }

            if (!args[2].isObject())
            {
                throw jsi::JSError(rt, "decodeBoxes: options must be an object");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            if (src.get() == dst.get())
            {
                throw jsi::JSError(rt, "decodeBoxes: In-place operations (src == dst) are not supported.");
            }
            auto opts = args[2].asObject(rt);

            if (!opts.hasProperty(rt, "from") || !opts.hasProperty(rt, "to"))
            {
                throw jsi::JSError(rt, "decodeBoxes: options must contain 'from' and 'to'");
            }

            auto fromVal = opts.getProperty(rt, "from");
            auto toVal = opts.getProperty(rt, "to");

            if (!fromVal.isString() || !toVal.isString())
            {
                throw jsi::JSError(rt, "decodeBoxes: options.from and options.to must be strings");
            }

            std::string from = fromVal.asString(rt).utf8(rt);
            std::string to = toVal.asString(rt).utf8(rt);

            if (from != "xyxy" && from != "xywh" && from != "cxcywh")
            {
                throw jsi::JSError(rt, "decodeBoxes: unsupported options.from format '" + from + "'");
            }

            if (to != "xyxy" && to != "xywh" && to != "cxcywh")
            {
                throw jsi::JSError(rt, "decodeBoxes: unsupported options.to format '" + to + "'");
            }

            if (src->shape_.size() != 2 || src->shape_[1] != 4)
            {
                throw jsi::JSError(rt, "decodeBoxes: src must be a 2D tensor with shape [N, 4]");
            }

            if (dst->shape_.size() != 2 || dst->shape_[1] != 4)
            {
                throw jsi::JSError(rt, "decodeBoxes: dst must be a 2D tensor with shape [N, 4]");
            }

            int N = src->shape_[0];
            if (dst->shape_[0] != N)
            {
                throw jsi::JSError(rt, "decodeBoxes: src and dst must have the same number of rows (N)");
            }

            if (src->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "decodeBoxes: src must have dtype float32");
            }

            if (dst->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "decodeBoxes: dst must have dtype float32");
            }

            // Lock src (shared) and dst (unique)
            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "decodeBoxes: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "decodeBoxes: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "decodeBoxes: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "decodeBoxes: dst tensor has been disposed");
            }

            const float *src_ptr = reinterpret_cast<const float *>(src->data_.get());
            float *dst_ptr = reinterpret_cast<float *>(dst->data_.get());

            if (from == to)
            {
                std::memcpy(dst_ptr, src_ptr, N * 4 * sizeof(float));
            }
            else
            {
                for (int i = 0; i < N; ++i)
                {
                    int offset = i * 4;
                    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;

                    if (from == "xyxy")
                    {
                        x1 = src_ptr[offset + 0];
                        y1 = src_ptr[offset + 1];
                        x2 = src_ptr[offset + 2];
                        y2 = src_ptr[offset + 3];
                    }
                    else if (from == "xywh")
                    {
                        x1 = src_ptr[offset + 0];
                        y1 = src_ptr[offset + 1];
                        x2 = x1 + src_ptr[offset + 2];
                        y2 = y1 + src_ptr[offset + 3];
                    }
                    else if (from == "cxcywh")
                    {
                        double cx = src_ptr[offset + 0];
                        double cy = src_ptr[offset + 1];
                        double w = src_ptr[offset + 2];
                        double h = src_ptr[offset + 3];
                        x1 = cx - w / 2.0;
                        y1 = cy - h / 2.0;
                        x2 = cx + w / 2.0;
                        y2 = cy + h / 2.0;
                    }

                    if (to == "xyxy")
                    {
                        dst_ptr[offset + 0] = static_cast<float>(x1);
                        dst_ptr[offset + 1] = static_cast<float>(y1);
                        dst_ptr[offset + 2] = static_cast<float>(x2);
                        dst_ptr[offset + 3] = static_cast<float>(y2);
                    }
                    else if (to == "xywh")
                    {
                        dst_ptr[offset + 0] = static_cast<float>(x1);
                        dst_ptr[offset + 1] = static_cast<float>(y1);
                        dst_ptr[offset + 2] = static_cast<float>(x2 - x1);
                        dst_ptr[offset + 3] = static_cast<float>(y2 - y1);
                    }
                    else if (to == "cxcywh")
                    {
                        dst_ptr[offset + 0] = static_cast<float>((x1 + x2) / 2.0);
                        dst_ptr[offset + 1] = static_cast<float>((y1 + y2) / 2.0);
                        dst_ptr[offset + 2] = static_cast<float>(x2 - x1);
                        dst_ptr[offset + 3] = static_cast<float>(y2 - y1);
                    }
                }
            }

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }

    void install_scaleBoxes(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "scaleBoxes";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 3)
            {
                throw jsi::JSError(rt, "Usage: scaleBoxes(src, dst, options)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "scaleBoxes: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "scaleBoxes: dst must be a Tensor");
            }

            if (!args[2].isObject())
            {
                throw jsi::JSError(rt, "scaleBoxes: options must be an object");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            if (src.get() == dst.get())
            {
                throw jsi::JSError(rt, "scaleBoxes: In-place operations (src == dst) are not supported.");
            }
            auto opts = args[2].asObject(rt);

            if (!opts.hasProperty(rt, "from") || !opts.hasProperty(rt, "to"))
            {
                throw jsi::JSError(rt, "scaleBoxes: options must contain 'from' and 'to'");
            }

            auto fromVal = opts.getProperty(rt, "from");
            auto toVal = opts.getProperty(rt, "to");

            if (!fromVal.isObject() || !fromVal.asObject(rt).isArray(rt) ||
                !toVal.isObject() || !toVal.asObject(rt).isArray(rt))
            {
                throw jsi::JSError(rt, "scaleBoxes: options.from and options.to must be arrays");
            }

            auto fromArr = fromVal.asObject(rt).asArray(rt);
            auto toArr = toVal.asObject(rt).asArray(rt);

            if (fromArr.length(rt) != 2 || toArr.length(rt) != 2)
            {
                throw jsi::JSError(rt, "scaleBoxes: options.from and options.to must have length 2 ([w, h])");
            }

            auto fromW = fromArr.getValueAtIndex(rt, 0);
            auto fromH = fromArr.getValueAtIndex(rt, 1);
            auto toW = toArr.getValueAtIndex(rt, 0);
            auto toH = toArr.getValueAtIndex(rt, 1);

            if (!fromW.isNumber() || !fromH.isNumber() || !toW.isNumber() || !toH.isNumber())
            {
                throw jsi::JSError(rt, "scaleBoxes: sizes must be numbers");
            }

            double scaleX = toW.asNumber() / fromW.asNumber();
            double scaleY = toH.asNumber() / fromH.asNumber();

            if (src->shape_.size() != 2 || src->shape_[1] != 4)
            {
                throw jsi::JSError(rt, "scaleBoxes: src must be a 2D tensor with shape [N, 4]");
            }

            if (dst->shape_.size() != 2 || dst->shape_[1] != 4)
            {
                throw jsi::JSError(rt, "scaleBoxes: dst must be a 2D tensor with shape [N, 4]");
            }

            int N = src->shape_[0];
            if (dst->shape_[0] != N)
            {
                throw jsi::JSError(rt, "scaleBoxes: src and dst must have the same number of rows (N)");
            }

            if (src->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "scaleBoxes: src must have dtype float32");
            }

            if (dst->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "scaleBoxes: dst must have dtype float32");
            }

            // Lock src (shared) and dst (unique)
            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "scaleBoxes: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "scaleBoxes: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "scaleBoxes: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "scaleBoxes: dst tensor has been disposed");
            }

            const float *src_ptr = reinterpret_cast<const float *>(src->data_.get());
            float *dst_ptr = reinterpret_cast<float *>(dst->data_.get());

            for (int i = 0; i < N; ++i)
            {
                int offset = i * 4;
                dst_ptr[offset + 0] = static_cast<float>(src_ptr[offset + 0] * scaleX);
                dst_ptr[offset + 1] = static_cast<float>(src_ptr[offset + 1] * scaleY);
                dst_ptr[offset + 2] = static_cast<float>(src_ptr[offset + 2] * scaleX);
                dst_ptr[offset + 3] = static_cast<float>(src_ptr[offset + 3] * scaleY);
            }

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }
} // namespace mylib::extensions::cv::box_ops
