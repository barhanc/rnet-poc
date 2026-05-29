#include "processing.h"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "../../core/Tensor.h"
#include "../../core/types.h"

namespace mylib::extensions::cv::processing
{
    namespace jsi = facebook::jsi;
    using TensorHostObject = mylib::core::tensor::TensorHostObject;

    static int interpToFlag(const std::string &interp)
    {
        if (interp == "nearest")
            return ::cv::INTER_NEAREST;
        if (interp == "area")
            return ::cv::INTER_AREA;
        if (interp == "cubic")
            return ::cv::INTER_CUBIC;
        if (interp == "lanczos")
            return ::cv::INTER_LANCZOS4;
        throw std::invalid_argument("resize: unsupported interpolation '" + interp + "'");
    }

    static int dtypeToCvDepth(mylib::core::types::DType dtype)
    {
        switch (dtype)
        {
        case mylib::core::types::DType::uint8:
            return CV_8U;
        case mylib::core::types::DType::int32:
            return CV_32S;
        case mylib::core::types::DType::float32:
            return CV_32F;
        }
        throw std::invalid_argument("resize: unsupported dtype");
    }

    void install_resize(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "resize";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 3)
            {
                throw jsi::JSError(rt, "Usage: resize(src, dst, options)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "resize: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "resize: dst must be a Tensor");
            }

            if (!args[2].isObject())
            {
                throw jsi::JSError(rt, "resize: options must be an object");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto opts = args[2].asObject(rt);

            if (!opts.hasProperty(rt, "mode") || !opts.getProperty(rt, "mode").isString())
            {
                throw jsi::JSError(rt, "resize: options.mode is required and must be a string");
            }

            if (!opts.hasProperty(rt, "interpolation") || !opts.getProperty(rt, "interpolation").isString())
            {
                throw jsi::JSError(rt, "resize: options.interpolation is required and must be a string");
            }

            if (!opts.hasProperty(rt, "padValue") || !opts.getProperty(rt, "padValue").isNumber())
            {
                throw jsi::JSError(rt, "resize: options.padValue is required and must be a number");
            }

            auto mode = opts.getProperty(rt, "mode").asString(rt).utf8(rt);
            auto interp = opts.getProperty(rt, "interpolation").asString(rt).utf8(rt);
            double padValue = opts.getProperty(rt, "padValue").asNumber();

            if (src->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "resize: src must be [H, W, C]");
            }

            if (dst->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "resize: dst must be [H, W, C]");
            }

            if (src->shape_[2] != dst->shape_[2])
            {
                throw jsi::JSError(rt, "resize: src and dst must have the same number of channels");
            }

            if (src->dtype_ != dst->dtype_)
            {
                throw jsi::JSError(rt, "resize: src and dst must have the same dtype");
            }

            // shared on src (read-only), unique on dst (write)

            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "resize: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "resize: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "resize: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "resize: dst tensor has been disposed");
            }

            int src_h = src->shape_[0];
            int src_w = src->shape_[1];
            int channels = src->shape_[2];
            int dst_h = dst->shape_[0];
            int dst_w = dst->shape_[1];

            int cv_type, interp_flag;
            try
            {
                cv_type = CV_MAKETYPE(dtypeToCvDepth(src->dtype_), channels);
                interp_flag = interpToFlag(interp);
            }
            catch (const std::invalid_argument &e)
            {
                throw jsi::JSError(rt, e.what());
            }

            ::cv::Mat src_mat(src_h, src_w, cv_type, src->data_.get());
            ::cv::Mat dst_mat(dst_h, dst_w, cv_type, dst->data_.get());

            if (mode == "stretch")
            {
                // Zero-alloc: cv::resize writes directly into dst->data_
                ::cv::resize(src_mat, dst_mat, dst_mat.size(), 0, 0, interp_flag);
            }
            else if (mode == "letterbox")
            {
                // Scale uniformly so src fits inside dst, pad remainder.
                // Zero-alloc: resize into an ROI submatrix view of dst_mat.
                double scale = std::min(static_cast<double>(dst_w) / src_w,
                                        static_cast<double>(dst_h) / src_h);

                int new_w = static_cast<int>(std::round(src_w * scale));
                int new_h = static_cast<int>(std::round(src_h * scale));
                int off_x = (dst_w - new_w) / 2;
                int off_y = (dst_h - new_h) / 2;

                dst_mat.setTo(::cv::Scalar::all(padValue));
                ::cv::Mat roi = dst_mat(::cv::Rect(off_x, off_y, new_w, new_h));
                ::cv::resize(src_mat, roi, roi.size(), 0, 0, interp_flag);
            }
            else if (mode == "crop")
            {
                // Scale so the *smaller* dimension fills dst, then center-crop.
                // Requires one temporary Mat because the scaled image is larger
                // than dst in at least one dimension.
                double scale = std::max(static_cast<double>(dst_w) / src_w,
                                        static_cast<double>(dst_h) / src_h);

                int new_w = static_cast<int>(std::round(src_w * scale));
                int new_h = static_cast<int>(std::round(src_h * scale));
                int off_x = (new_w - dst_w) / 2;
                int off_y = (new_h - dst_h) / 2;

                ::cv::Mat scaled;
                ::cv::resize(src_mat, scaled, ::cv::Size(new_w, new_h), 0, 0, interp_flag);
                scaled(::cv::Rect(off_x, off_y, dst_w, dst_h)).copyTo(dst_mat);
            }
            else
            {
                throw jsi::JSError(rt, "resize: unknown mode '" + mode + "'. Use 'stretch', 'letterbox', or 'crop'");
            }

            return jsi::Value::undefined();
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }

} // namespace mylib::extensions::cv::processing
