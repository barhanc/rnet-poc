#include "processing.h"

#include <cmath>
#include <stdexcept>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "core/tensor.h"
#include "core/types.h"

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

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }

    static int codeToColorConversionFlag(const std::string &code)
    {
        if (code == "RGBA2RGB")
            return ::cv::COLOR_RGBA2RGB;
        if (code == "RGBA2BGR")
            return ::cv::COLOR_RGBA2BGR;
        if (code == "BGRA2RGBA")
            return ::cv::COLOR_BGRA2RGBA;
        if (code == "BGRA2RGB")
            return ::cv::COLOR_BGRA2RGB;
        if (code == "BGRA2BGR")
            return ::cv::COLOR_BGRA2BGR;
        if (code == "RGB2BGR")
            return ::cv::COLOR_RGB2BGR;
        if (code == "BGR2RGB")
            return ::cv::COLOR_BGR2RGB;
        if (code == "RGB2RGBA")
            return ::cv::COLOR_RGB2RGBA;
        if (code == "BGR2RGBA")
            return ::cv::COLOR_BGR2RGBA;
        if (code == "RGB2GRAY")
            return ::cv::COLOR_RGB2GRAY;
        if (code == "RGBA2GRAY")
            return ::cv::COLOR_RGBA2GRAY;
        if (code == "BGR2GRAY")
            return ::cv::COLOR_BGR2GRAY;
        if (code == "BGRA2GRAY")
            return ::cv::COLOR_BGRA2GRAY;
        throw std::invalid_argument("cvtColor: unsupported color conversion code '" + code + "'");
    }

    void install_cvtColor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "cvtColor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 3)
            {
                throw jsi::JSError(rt, "Usage: cvtColor(src, dst, code)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "cvtColor: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "cvtColor: dst must be a Tensor");
            }

            if (!args[2].isString())
            {
                throw jsi::JSError(rt, "cvtColor: code must be a string");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto code = args[2].asString(rt).utf8(rt);

            if (src->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "cvtColor: src must be a 3D tensor [H, W, C]");
            }

            if (dst->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "cvtColor: dst must be a 3D tensor [H, W, C]");
            }

            if (src->shape_[0] != dst->shape_[0] || src->shape_[1] != dst->shape_[1])
            {
                throw jsi::JSError(rt, "cvtColor: src and dst spatial dimensions (H, W) must match");
            }

            if (src->dtype_ != dst->dtype_)
            {
                throw jsi::JSError(rt, "cvtColor: src and dst must have the same dtype");
            }

            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "cvtColor: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "cvtColor: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "cvtColor: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "cvtColor: dst tensor has been disposed");
            }

            int src_h = src->shape_[0];
            int src_w = src->shape_[1];
            int src_c = src->shape_[2];
            int dst_c = dst->shape_[2];

            int cv_src_type, cv_dst_type, flag;
            try
            {
                cv_src_type = CV_MAKETYPE(dtypeToCvDepth(src->dtype_), src_c);
                cv_dst_type = CV_MAKETYPE(dtypeToCvDepth(dst->dtype_), dst_c);
                flag = codeToColorConversionFlag(code);

                ::cv::Mat src_mat(src_h, src_w, cv_src_type, src->data_.get());
                ::cv::Mat dst_mat(src_h, src_w, cv_dst_type, dst->data_.get());

                ::cv::cvtColor(src_mat, dst_mat, flag);
            }
            catch (const std::invalid_argument &e)
            {
                throw jsi::JSError(rt, e.what());
            }

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }

    void install_toChannelsFirst(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "toChannelsFirst";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: toChannelsFirst(src, dst)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "toChannelsFirst: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "toChannelsFirst: dst must be a Tensor");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            if (src->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "toChannelsFirst: src must be a 3D tensor [H, W, C]");
            }

            if (src->dtype_ != dst->dtype_)
            {
                throw jsi::JSError(rt, "toChannelsFirst: src and dst must have the same dtype");
            }

            int src_h = src->shape_[0];
            int src_w = src->shape_[1];
            int src_c = src->shape_[2];

            if (dst->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "toChannelsFirst: dst must be a 3D tensor [C, H, W]");
            }
            int dst_c = dst->shape_[0];
            int dst_h = dst->shape_[1];
            int dst_w = dst->shape_[2];

            if (src_h != dst_h || src_w != dst_w || src_c != dst_c)
            {
                throw jsi::JSError(rt, "toChannelsFirst: src and dst spatial dimensions and channel counts must match");
            }

            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "toChannelsFirst: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "toChannelsFirst: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "toChannelsFirst: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "toChannelsFirst: dst tensor has been disposed");
            }

            int cv_type;
            try
            {
                cv_type = CV_MAKETYPE(dtypeToCvDepth(src->dtype_), src_c);
            }
            catch (const std::invalid_argument &e)
            {
                throw jsi::JSError(rt, e.what());
            }

            ::cv::Mat src_mat(src_h, src_w, cv_type, src->data_.get());
            std::vector<::cv::Mat> channels;
            ::cv::split(src_mat, channels);

            int hw = src_h * src_w;
            size_t elem_size = mylib::core::types::elementSize(src->dtype_);
            uint8_t *dst_ptr = dst->data_.get();

            for (int i = 0; i < src_c; ++i)
            {
                std::memcpy(dst_ptr + i * hw * elem_size, channels[i].data, hw * elem_size);
            }

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody));
    }

    void install_toChannelsLast(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "toChannelsLast";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: toChannelsLast(src, dst)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "toChannelsLast: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "toChannelsLast: dst must be a Tensor");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            if (src->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "toChannelsLast: src must be a 3D tensor [C, H, W]");
            }

            if (src->dtype_ != dst->dtype_)
            {
                throw jsi::JSError(rt, "toChannelsLast: src and dst must have the same dtype");
            }

            int src_c = src->shape_[0];
            int src_h = src->shape_[1];
            int src_w = src->shape_[2];

            if (dst->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "toChannelsLast: dst must be a 3D tensor [H, W, C]");
            }
            int dst_h = dst->shape_[0];
            int dst_w = dst->shape_[1];
            int dst_c = dst->shape_[2];

            if (src_h != dst_h || src_w != dst_w || src_c != dst_c)
            {
                throw jsi::JSError(rt, "toChannelsLast: src and dst spatial dimensions and channel counts must match");
            }

            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "toChannelsLast: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "toChannelsLast: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "toChannelsLast: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "toChannelsLast: dst tensor has been disposed");
            }

            int cv_depth;
            try
            {
                cv_depth = dtypeToCvDepth(src->dtype_);
            }
            catch (const std::invalid_argument &e)
            {
                throw jsi::JSError(rt, e.what());
            }

            int hw = src_h * src_w;
            size_t elem_size = mylib::core::types::elementSize(src->dtype_);
            uint8_t *src_ptr = src->data_.get();

            std::vector<::cv::Mat> channels;
            for (int i = 0; i < src_c; ++i)
            {
                channels.push_back(::cv::Mat(src_h, src_w, cv_depth, src_ptr + i * hw * elem_size));
            }

            ::cv::Mat dst_mat(dst_h, dst_w, CV_MAKETYPE(cv_depth, dst_c), dst->data_.get());
            ::cv::merge(channels, dst_mat);

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody));
    }

    void install_normalize(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "normalize";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 3)
            {
                throw jsi::JSError(rt, "Usage: normalize(src, dst, options)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "normalize: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "normalize: dst must be a Tensor");
            }

            if (!args[2].isObject())
            {
                throw jsi::JSError(rt, "normalize: options must be an object");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto opts = args[2].asObject(rt);

            if (src->shape_.size() != 3)
            {
                throw jsi::JSError(rt, "normalize: src must be a 3D tensor [C, H, W]");
            }

            int c = src->shape_[0];
            int h = src->shape_[1];
            int w = src->shape_[2];

            bool dstMatch = false;
            if (dst->shape_.size() == 3 &&
                dst->shape_[0] == c &&
                dst->shape_[1] == h &&
                dst->shape_[2] == w)
            {
                dstMatch = true;
            }

            if (!dstMatch)
            {
                throw jsi::JSError(rt, "normalize: src and dst shapes must match exactly ([C, H, W])");
            }

            if (!opts.hasProperty(rt, "alpha"))
            {
                throw jsi::JSError(rt, "normalize: options.alpha is required");
            }

            if (!opts.hasProperty(rt, "beta"))
            {
                throw jsi::JSError(rt, "normalize: options.beta is required");
            }

            std::vector<double> alpha(c, 1.0); // default to no scaling
            std::vector<double> beta(c, 0.0);  // default to no shifting

            auto alphaVal = opts.getProperty(rt, "alpha");
            if (alphaVal.isNumber())
            {
                std::fill(alpha.begin(), alpha.end(), alphaVal.asNumber());
            }
            else if (alphaVal.isObject() && alphaVal.asObject(rt).isArray(rt))
            {
                auto arr = alphaVal.asObject(rt).asArray(rt);
                if (arr.length(rt) != static_cast<size_t>(c))
                {
                    throw jsi::JSError(rt, "normalize: options.alpha array length must be exactly equal to channels");
                }
                for (int i = 0; i < c; ++i)
                {
                    auto val = arr.getValueAtIndex(rt, i);
                    if (!val.isNumber())
                    {
                        throw jsi::JSError(rt, "normalize: options.alpha array must contain only numbers");
                    }
                    alpha[i] = val.asNumber();
                }
            }
            else
            {
                throw jsi::JSError(rt, "normalize: options.alpha must be a number or an array of numbers");
            }

            auto betaVal = opts.getProperty(rt, "beta");
            if (betaVal.isNumber())
            {
                std::fill(beta.begin(), beta.end(), betaVal.asNumber());
            }
            else if (betaVal.isObject() && betaVal.asObject(rt).isArray(rt))
            {
                auto arr = betaVal.asObject(rt).asArray(rt);
                if (arr.length(rt) != static_cast<size_t>(c))
                {
                    throw jsi::JSError(rt, "normalize: options.beta array length must be exactly equal to channels");
                }
                for (int i = 0; i < c; ++i)
                {
                    auto val = arr.getValueAtIndex(rt, i);
                    if (!val.isNumber())
                    {
                        throw jsi::JSError(rt, "normalize: options.beta array must contain only numbers");
                    }
                    beta[i] = val.asNumber();
                }
            }
            else
            {
                throw jsi::JSError(rt, "normalize: options.beta must be a number or an array of numbers");
            }

            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "normalize: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "normalize: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "normalize: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "normalize: dst tensor has been disposed");
            }

            int src_depth_type;
            int dst_depth_type;
            try
            {
                src_depth_type = dtypeToCvDepth(src->dtype_);
                dst_depth_type = dtypeToCvDepth(dst->dtype_);
            }
            catch (const std::invalid_argument &e)
            {
                throw jsi::JSError(rt, e.what());
            }

            size_t src_elem_size = mylib::core::types::elementSize(src->dtype_);
            size_t dst_elem_size = mylib::core::types::elementSize(dst->dtype_);
            uint8_t *src_ptr = src->data_.get();
            uint8_t *dst_ptr = dst->data_.get();

            for (int ch = 0; ch < c; ++ch)
            {
                ::cv::Mat src_channel(h, w, src_depth_type, src_ptr + ch * h * w * src_elem_size);
                ::cv::Mat dst_channel(h, w, dst_depth_type, dst_ptr + ch * h * w * dst_elem_size);

                src_channel.convertTo(dst_channel, dst_depth_type, alpha[ch], beta[ch]);
            }

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }

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

} // namespace mylib::extensions::cv::processing
