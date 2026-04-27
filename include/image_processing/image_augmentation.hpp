#ifndef IMAGE_AUGMENTATION_HPP
#define IMAGE_AUGMENTATION_HPP

#include "tensors/tensor.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

namespace neural {

class ImageAugmentation
{
	public:
		ImageAugmentation( std::size_t rows, std::size_t cols, std::size_t channels ) noexcept;
		virtual ~ImageAugmentation() = default;
		virtual void apply( Tensor<float> &in, Tensor<float> &out ) const = 0;
	protected:
		std::size_t m_rows;
		std::size_t m_cols;
		std::size_t m_channels;
};

class ImageAugmentationCombinatorial : public ImageAugmentation
{
	public:
		ImageAugmentationCombinatorial(
		    std::size_t rows, std::size_t cols, std::size_t channels, float p,
		    std::vector<std::unique_ptr<ImageAugmentation>> augmentations );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;

	private:
		/// Bernoulli probability per stage that the corresponding sub-augmentation runs.
		float m_p;
		std::vector<std::unique_ptr<ImageAugmentation>> m_augmentations;
		mutable std::mt19937 m_rng;
};

class RandomRotation: public ImageAugmentation
{
	public:
		RandomRotation( std::size_t rows, std::size_t cols, std::size_t channels,
		                float max_angle_deg = 15.0f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_max_angle_deg;
		mutable std::mt19937 m_rng;
};

class RandomCrop: public ImageAugmentation
{
	public:
		RandomCrop( std::size_t rows, std::size_t cols, std::size_t channels,
		            float min_scale = 0.8f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_min_scale;
		mutable std::mt19937 m_rng;
};

/// Left–right flip with probability \a p (default 0.5).
class RandomHorizontalFlip: public ImageAugmentation
{
	public:
		RandomHorizontalFlip( std::size_t rows, std::size_t cols, std::size_t channels,
		                      float p = 0.5f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_p;
		mutable std::mt19937 m_rng;
};

class Affine: public ImageAugmentation
{
	public:
		Affine( std::size_t rows, std::size_t cols, std::size_t channels, float max_rot_deg = 10.0f,
		        float max_trans_frac = 0.05f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_max_rot_deg;
		float m_max_trans_frac;
		mutable std::mt19937 m_rng;
};

class ColorJitter: public ImageAugmentation
{
	public:
		ColorJitter( std::size_t rows, std::size_t cols, std::size_t channels,
		             float brightness = 0.1f, float contrast = 0.1f, float saturation = 0.1f,
		             float hue_deg = 10.0f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_brightness;
		float m_contrast;
		float m_saturation;
		float m_hue_deg;
		mutable std::mt19937 m_rng;
};

class GaussianBlur: public ImageAugmentation
{
	public:
		GaussianBlur( std::size_t rows, std::size_t cols, std::size_t channels, int ksize = 3,
		              double sigma = 0.0 );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		int m_ksize;
		double m_sigma;
};

class Noise: public ImageAugmentation
{
	public:
		Noise( std::size_t rows, std::size_t cols, std::size_t channels, float sigma = 0.02f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_sigma;
		mutable std::mt19937 m_rng;
};

class Posterize: public ImageAugmentation
{
	public:
		Posterize( std::size_t rows, std::size_t cols, std::size_t channels, int levels = 6 );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		int m_levels;
};

class Solarize: public ImageAugmentation
{
	public:
		Solarize( std::size_t rows, std::size_t cols, std::size_t channels, float threshold = 0.5f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_threshold;
};

class Cutout: public ImageAugmentation
{
	public:
		Cutout( std::size_t rows, std::size_t cols, std::size_t channels, float hole_frac = 0.25f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_hole_frac;
		mutable std::mt19937 m_rng;
};

class RandomErasing: public ImageAugmentation
{
	public:
		RandomErasing( std::size_t rows, std::size_t cols, std::size_t channels,
		               float area_frac = 0.1f, float aspect_min = 0.3f, float aspect_max = 3.3f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		float m_area_frac;
		float m_aspect_min;
		float m_aspect_max;
		mutable std::mt19937 m_rng;
};

class HideAndSeek: public ImageAugmentation
{
	public:
		HideAndSeek( std::size_t rows, std::size_t cols, std::size_t channels, int num_patches = 4,
		             float patch_frac = 0.12f );
		void apply( Tensor<float> &in, Tensor<float> &out ) const override;
	private:
		int m_num_patches;
		float m_patch_frac;
		mutable std::mt19937 m_rng;
};

/// CIFAR-10: 32×32, 3 channels, planar R|G|B (same layout as \c cifar10_input_dim in \c cifar10_loader.hpp).
inline constexpr std::size_t cifar10_augment_rows   = 32;
inline constexpr std::size_t cifar10_augment_cols   = 32;
inline constexpr std::size_t cifar10_augment_ch     = 3;
inline constexpr std::size_t cifar10_augment_flat   = cifar10_augment_rows * cifar10_augment_cols * cifar10_augment_ch;

/// Training-time pipeline for CIFAR-10 matching the cifar10vgg.py reference:
/// random horizontal flip (p=0.5) + affine (±15° rotation, ±10% translation), always applied.
[[nodiscard]] std::unique_ptr<ImageAugmentation> make_cifar10_training_augmentation( float p = 1.0f );

/// Compute the global scalar mean and standard deviation of \a images (all pixels, all channels)
/// using OpenCV \c meanStdDev. Returns \c {mean, std}.
std::pair<float, float> cifar10_compute_normalization(
    std::vector<Tensor<float>> const &images );

/// Normalize \a images in-place: x ← (x − \a mean) / (\a std + eps).
void cifar10_apply_normalization( std::vector<Tensor<float>> &images, float mean, float std,
                                  float eps = 1e-7f );

/// Use with \c MomentumSGDOptimizer::train(..., op) / \c SGDOptimizer::train(..., op): after \c assignBatch
/// fills \a host_x (images) and \a host_y (targets), rewrites \a host_x in-place: optional augmentation, then
/// scalar normalization (same as \ref cifar10_apply_normalization) on each row so **training** can keep host
/// images in [0,1] while **eval** uses \c cifar10_apply_normalization on stored test images only.
/// \a input_cols must equal \c cifar10_augment_flat.
struct Cifar10AugmentBatchOp
{
	std::size_t input_cols  = 0;
	std::size_t target_cols = 0;
	std::size_t batch_rows  = 0;
	std::vector<std::unique_ptr<ImageAugmentation>> augmentations;
	mutable std::vector<Tensor<float>> tmp_in;
	mutable std::vector<Tensor<float>> tmp_out;
	float m_norm_mean  = 0.f;
	/// 0: skip; else multiply by this after \c (x - m_norm_mean) (i.e. \c 1/(std+eps)).
	float m_norm_inv   = 0.f;

	Cifar10AugmentBatchOp() = default;
	Cifar10AugmentBatchOp( std::size_t in_cols, std::size_t out_cols, std::size_t batch_rows,
	                       std::vector<std::unique_ptr<ImageAugmentation>> &&augmentations,
	                       float norm_mean, float norm_std, float norm_eps = 1e-7f );

	void operator()( Tensor<float> &host_x, Tensor<float> &host_y ) const;
};

} // namespace neural

#endif
