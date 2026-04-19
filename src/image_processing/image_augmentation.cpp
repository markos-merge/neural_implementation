#include "image_processing/image_augmentation.hpp"

#include "cifar10_loader.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neural {

namespace {

std::size_t flat_dim( std::size_t rows, std::size_t cols, std::size_t ch )
{
	return rows * cols * ch;
}

/// Planar layout: ch0 plane, ch1 plane, … (CIFAR-style RGB = R, then G, then B).
void tensor_row_to_mat( float const *row, std::size_t h, std::size_t w, std::size_t ch, cv::Mat &out )
{
	if ( ch == 1 ) {
		cv::Mat( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1,
		         const_cast<float *>( row ) )
		    .copyTo( out );
		return;
	}
	if ( ch == 3 ) {
		float const *p = row;
		cv::Mat const R( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1,
		                 const_cast<float *>( p ) );
		cv::Mat const G( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1,
		                 const_cast<float *>( p + h * w ) );
		cv::Mat const B( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1,
		                 const_cast<float *>( p + 2 * h * w ) );
		std::vector<cv::Mat> const planes = { B, G, R };
		cv::merge( planes, out );
		return;
	}
	std::vector<cv::Mat> planes;
	planes.reserve( ch );
	for ( std::size_t c = 0; c < ch; ++c ) {
		float const *pc = row + c * h * w;
		planes.emplace_back( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1,
		                     const_cast<float *>( pc ) );
	}
	cv::merge( planes, out );
}

void mat_to_tensor_row_planar( cv::Mat const &m, float *row, std::size_t h, std::size_t w,
                               std::size_t ch )
{
	if ( m.type() != CV_32FC( static_cast<int>( ch ) ) ) {
		throw std::runtime_error( "image_augmentation: unexpected cv::Mat type after transform" );
	}
	if ( ch == 1 ) {
		m.copyTo( cv::Mat( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1, row ) );
		return;
	}
	if ( ch == 3 ) {
		std::vector<cv::Mat> planes;
		cv::split( m, planes );
		planes[2].copyTo( cv::Mat( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1, row ) );
		planes[1].copyTo(
		    cv::Mat( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1, row + h * w ) );
		planes[0].copyTo(
		    cv::Mat( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1, row + 2 * h * w ) );
		return;
	}
	std::vector<cv::Mat> planes;
	cv::split( m, planes );
	for ( std::size_t c = 0; c < ch; ++c ) {
		planes[c].copyTo( cv::Mat( static_cast<int>( h ), static_cast<int>( w ), CV_32FC1,
		                  row + c * h * w ) );
	}
}

void clamp01( cv::Mat &m )
{
	CV_Assert( m.depth() == CV_32F );
	int const n = m.rows * m.cols * m.channels();
	float *p = m.ptr<float>( 0 );
	for ( int i = 0; i < n; ++i ) {
		p[i] = std::clamp( p[i], 0.0f, 1.0f );
	}
}

} // namespace

ImageAugmentation::ImageAugmentation( std::size_t rows, std::size_t cols,
                                      std::size_t channels ) noexcept
    : m_rows( rows )
    , m_cols( cols )
    , m_channels( channels )
{
}

ImageAugmentationCombinatorial::ImageAugmentationCombinatorial(
    std::size_t rows, std::size_t cols, std::size_t channels, float p,
    std::vector<std::unique_ptr<ImageAugmentation>> augmentations )
    : ImageAugmentation( rows, cols, channels ),
		m_p( p ),
    m_augmentations( std::move( augmentations ) ),
		m_rng( std::random_device{}() )
{
}

void ImageAugmentationCombinatorial::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "ImageAugmentationCombinatorial: tensor width mismatch" );
	}
	if ( m_augmentations.empty() ) {
		out = in;
		return;
	}
	Tensor<float> cur( in );
	std::bernoulli_distribution try_aug(
	    static_cast<double>( std::clamp( m_p, 0.0f, 1.0f ) ) );
	for ( std::size_t i = 0; i < m_augmentations.size(); ++i ) {
		if ( !try_aug( m_rng ) ) {
			continue;
		}
		Tensor<float> nxt( in.rows(), in.cols() );
		m_augmentations[i]->apply( cur, nxt );
		cur = std::move( nxt );
	}
	out = std::move( cur );
}

RandomRotation::RandomRotation( std::size_t rows, std::size_t cols, std::size_t channels,
                                float max_angle_deg )
    : ImageAugmentation( rows, cols, channels )
    , m_max_angle_deg( max_angle_deg )
    , m_rng( std::random_device{}() )
{
}

void RandomRotation::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "RandomRotation: tensor shape mismatch" );
	}
	std::uniform_real_distribution<float> ang( -m_max_angle_deg, m_max_angle_deg );
	cv::Point2f const center( static_cast<float>( m_cols - 1 ) * 0.5f,
	                          static_cast<float>( m_rows - 1 ) * 0.5f );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		cv::Mat M = cv::getRotationMatrix2D( center, static_cast<double>( ang( m_rng ) ), 1.0 );
		cv::Mat warped;
		cv::warpAffine( img, warped, M, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101 );
		clamp01( warped );
		mat_to_tensor_row_planar( warped, out_row, m_rows, m_cols, m_channels );
	}
}

RandomCrop::RandomCrop( std::size_t rows, std::size_t cols, std::size_t channels, float min_scale )
    : ImageAugmentation( rows, cols, channels )
    , m_min_scale( min_scale )
    , m_rng( std::random_device{}() )
{
}

void RandomCrop::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "RandomCrop: tensor shape mismatch" );
	}
	std::uniform_real_distribution<float> scale_dist( m_min_scale, 1.0f );
	std::uniform_int_distribution<int> x_dist;
	std::uniform_int_distribution<int> y_dist;
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		float const s = scale_dist( m_rng );
		int const crop_h = std::max( 1, static_cast<int>( std::lround( s * static_cast<float>( m_rows ) ) ) );
		int const crop_w = std::max( 1, static_cast<int>( std::lround( s * static_cast<float>( m_cols ) ) ) );
		x_dist.param( std::uniform_int_distribution<int>::param_type( 0, std::max( 0, static_cast<int>( m_cols ) - crop_w ) ) );
		y_dist.param( std::uniform_int_distribution<int>::param_type( 0, std::max( 0, static_cast<int>( m_rows ) - crop_h ) ) );
		int const x0 = x_dist( m_rng );
		int const y0 = y_dist( m_rng );
		cv::Mat roi = img( cv::Rect( x0, y0, crop_w, crop_h ) );
		cv::Mat resized;
		cv::resize( roi, resized, cv::Size( static_cast<int>( m_cols ), static_cast<int>( m_rows ) ), 0, 0,
		            cv::INTER_LINEAR );
		clamp01( resized );
		mat_to_tensor_row_planar( resized, out_row, m_rows, m_cols, m_channels );
	}
}

RandomHorizontalFlip::RandomHorizontalFlip( std::size_t rows, std::size_t cols,
                                              std::size_t channels, float p )
    : ImageAugmentation( rows, cols, channels )
    , m_p( std::clamp( p, 0.0f, 1.0f ) )
    , m_rng( std::random_device{}() )
{
}

void RandomHorizontalFlip::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "RandomHorizontalFlip: tensor shape mismatch" );
	}
	std::bernoulli_distribution flip( static_cast<double>( m_p ) );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		cv::Mat out_img;
		if ( flip( m_rng ) ) {
			cv::flip( img, out_img, 1 );
		} else {
			img.copyTo( out_img );
		}
		mat_to_tensor_row_planar( out_img, out_row, m_rows, m_cols, m_channels );
	}
}

Affine::Affine( std::size_t rows, std::size_t cols, std::size_t channels, float max_rot_deg,
                float max_trans_frac )
    : ImageAugmentation( rows, cols, channels )
    , m_max_rot_deg( max_rot_deg )
    , m_max_trans_frac( max_trans_frac )
    , m_rng( std::random_device{}() )
{
}

void Affine::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "Affine: tensor shape mismatch" );
	}
	std::uniform_real_distribution<float> rot( -m_max_rot_deg, m_max_rot_deg );
	std::uniform_real_distribution<float> tr(
	    -m_max_trans_frac * static_cast<float>( std::max( m_rows, m_cols ) ),
	    m_max_trans_frac * static_cast<float>( std::max( m_rows, m_cols ) ) );
	cv::Point2f const center( static_cast<float>( m_cols - 1 ) * 0.5f,
	                          static_cast<float>( m_rows - 1 ) * 0.5f );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		cv::Mat M = cv::getRotationMatrix2D( center, static_cast<double>( rot( m_rng ) ), 1.0 );
		M.at<double>( 0, 2 ) += static_cast<double>( tr( m_rng ) );
		M.at<double>( 1, 2 ) += static_cast<double>( tr( m_rng ) );
		cv::Mat warped;
		cv::warpAffine( img, warped, M, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101 );
		clamp01( warped );
		mat_to_tensor_row_planar( warped, out_row, m_rows, m_cols, m_channels );
	}
}

ColorJitter::ColorJitter( std::size_t rows, std::size_t cols, std::size_t channels,
                          float brightness, float contrast, float saturation, float hue_deg )
    : ImageAugmentation( rows, cols, channels )
    , m_brightness( brightness )
    , m_contrast( contrast )
    , m_saturation( saturation )
    , m_hue_deg( hue_deg )
    , m_rng( std::random_device{}() )
{
}

void ColorJitter::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "ColorJitter: tensor shape mismatch" );
	}
	std::uniform_real_distribution<float> ub( -m_brightness, m_brightness );
	std::uniform_real_distribution<float> uc( 1.0f - m_contrast, 1.0f + m_contrast );
	std::uniform_real_distribution<float> us( 1.0f - m_saturation, 1.0f + m_saturation );
	std::uniform_real_distribution<float> uh( -m_hue_deg, m_hue_deg );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		float const b = ub( m_rng );
		float const c = uc( m_rng );
		img = img * c + b;
		if ( m_channels == 3 ) {
			cv::Mat u8;
			clamp01( img );
			img.convertTo( u8, CV_8UC3, 255.0 );
			cv::Mat hsv;
			cv::cvtColor( u8, hsv, cv::COLOR_BGR2HSV );
			float const dh = uh( m_rng );
			float const ds = us( m_rng );
			for ( int y = 0; y < hsv.rows; ++y ) {
				cv::Vec3b *py = hsv.ptr<cv::Vec3b>( y );
				for ( int x = 0; x < hsv.cols; ++x ) {
					int h = static_cast<int>( py[x][0] ) + static_cast<int>( std::lround( dh ) );
					h = ( h % 180 + 180 ) % 180;
					py[x][0] = static_cast<unsigned char>( h );
					int s = static_cast<int>( py[x][1] );
					s = static_cast<int>( std::round( static_cast<float>( s ) * ds ) );
					s = std::clamp( s, 0, 255 );
					py[x][1] = static_cast<unsigned char>( s );
				}
			}
			cv::cvtColor( hsv, u8, cv::COLOR_HSV2BGR );
			u8.convertTo( img, CV_32FC3, 1.0 / 255.0 );
		}
		clamp01( img );
		mat_to_tensor_row_planar( img, out_row, m_rows, m_cols, m_channels );
	}
}

GaussianBlur::GaussianBlur( std::size_t rows, std::size_t cols, std::size_t channels, int ksize,
                            double sigma )
    : ImageAugmentation( rows, cols, channels )
    , m_ksize( ksize | 1 )
    , m_sigma( sigma )
{
	if ( m_ksize < 1 ) {
		m_ksize = 1;
	}
}

void GaussianBlur::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "GaussianBlur: tensor shape mismatch" );
	}
	int const k = m_ksize | 1;
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		cv::Mat blurred;
		cv::GaussianBlur( img, blurred, cv::Size( k, k ), m_sigma, m_sigma, cv::BORDER_DEFAULT );
		clamp01( blurred );
		mat_to_tensor_row_planar( blurred, out_row, m_rows, m_cols, m_channels );
	}
}

Noise::Noise( std::size_t rows, std::size_t cols, std::size_t channels, float sigma )
    : ImageAugmentation( rows, cols, channels )
    , m_sigma( sigma )
    , m_rng( std::random_device{}() )
{
}

void Noise::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "Noise: tensor shape mismatch" );
	}
	std::normal_distribution<float> dist( 0.0f, m_sigma );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		for ( int y = 0; y < img.rows; ++y ) {
			float *py = img.ptr<float>( y );
			for ( int x = 0; x < img.cols * img.channels(); ++x ) {
				py[x] += dist( m_rng );
			}
		}
		clamp01( img );
		mat_to_tensor_row_planar( img, out_row, m_rows, m_cols, m_channels );
	}
}

Posterize::Posterize( std::size_t rows, std::size_t cols, std::size_t channels, int levels )
    : ImageAugmentation( rows, cols, channels )
    , m_levels( std::max( 2, levels ) )
{
}

void Posterize::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "Posterize: tensor shape mismatch" );
	}
	float const L = static_cast<float>( m_levels - 1 );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		for ( int y = 0; y < img.rows; ++y ) {
			float *py = img.ptr<float>( y );
			for ( int x = 0; x < img.cols * img.channels(); ++x ) {
				float v = py[x];
				v = std::floor( v * L + 0.5f ) / L;
				py[x] = v;
			}
		}
		clamp01( img );
		mat_to_tensor_row_planar( img, out_row, m_rows, m_cols, m_channels );
	}
}

Solarize::Solarize( std::size_t rows, std::size_t cols, std::size_t channels, float threshold )
    : ImageAugmentation( rows, cols, channels )
    , m_threshold( threshold )
{
}

void Solarize::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "Solarize: tensor shape mismatch" );
	}
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		for ( int y = 0; y < img.rows; ++y ) {
			float *py = img.ptr<float>( y );
			for ( int x = 0; x < img.cols * img.channels(); ++x ) {
				float v = py[x];
				if ( v >= m_threshold ) {
					v = 1.0f - v;
				}
				py[x] = v;
			}
		}
		clamp01( img );
		mat_to_tensor_row_planar( img, out_row, m_rows, m_cols, m_channels );
	}
}

Cutout::Cutout( std::size_t rows, std::size_t cols, std::size_t channels, float hole_frac )
    : ImageAugmentation( rows, cols, channels )
    , m_hole_frac( hole_frac )
    , m_rng( std::random_device{}() )
{
}

void Cutout::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "Cutout: tensor shape mismatch" );
	}
	int const side = std::max(
	    1, static_cast<int>( std::lround( m_hole_frac * static_cast<float>( std::min( m_rows, m_cols ) ) ) ) );
	std::uniform_int_distribution<int> x_dist( 0, std::max( 0, static_cast<int>( m_cols ) - side ) );
	std::uniform_int_distribution<int> y_dist( 0, std::max( 0, static_cast<int>( m_rows ) - side ) );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		int const x0 = x_dist( m_rng );
		int const y0 = y_dist( m_rng );
		img( cv::Rect( x0, y0, side, side ) ) = 0.0f;
		mat_to_tensor_row_planar( img, out_row, m_rows, m_cols, m_channels );
	}
}

RandomErasing::RandomErasing( std::size_t rows, std::size_t cols, std::size_t channels,
                              float area_frac, float aspect_min, float aspect_max )
    : ImageAugmentation( rows, cols, channels )
    , m_area_frac( area_frac )
    , m_aspect_min( aspect_min )
    , m_aspect_max( aspect_max )
    , m_rng( std::random_device{}() )
{
}

void RandomErasing::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "RandomErasing: tensor shape mismatch" );
	}
	std::uniform_real_distribution<float> uni( 0.0f, 1.0f );
	float const area = m_area_frac * static_cast<float>( m_rows * m_cols );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		float ar = std::exp( uni( m_rng ) * ( std::log( m_aspect_max ) - std::log( m_aspect_min ) )
		                     + std::log( m_aspect_min ) );
		int h = std::max( 1, static_cast<int>( std::sqrt( area / ar ) ) );
		int w = std::max( 1, static_cast<int>( area / static_cast<float>( h ) ) );
		h = std::min( h, static_cast<int>( m_rows ) );
		w = std::min( w, static_cast<int>( m_cols ) );
		std::uniform_int_distribution<int> x_dist( 0, std::max( 0, static_cast<int>( m_cols ) - w ) );
		std::uniform_int_distribution<int> y_dist( 0, std::max( 0, static_cast<int>( m_rows ) - h ) );
		int const x0 = x_dist( m_rng );
		int const y0 = y_dist( m_rng );
		img( cv::Rect( x0, y0, w, h ) ) = 0.0f;
		mat_to_tensor_row_planar( img, out_row, m_rows, m_cols, m_channels );
	}
}

HideAndSeek::HideAndSeek( std::size_t rows, std::size_t cols, std::size_t channels,
                          int num_patches, float patch_frac )
    : ImageAugmentation( rows, cols, channels )
    , m_num_patches( std::max( 1, num_patches ) )
    , m_patch_frac( patch_frac )
    , m_rng( std::random_device{}() )
{
}

void HideAndSeek::apply( Tensor<float> &in, Tensor<float> &out ) const
{
	std::size_t const flat = flat_dim( m_rows, m_cols, m_channels );
	if ( in.cols() != out.cols() || in.cols() != flat ) {
		throw std::invalid_argument( "HideAndSeek: tensor shape mismatch" );
	}
	int const side = std::max(
	    1, static_cast<int>( std::lround( m_patch_frac * static_cast<float>( std::min( m_rows, m_cols ) ) ) ) );
	std::uniform_int_distribution<int> x_dist( 0, std::max( 0, static_cast<int>( m_cols ) - side ) );
	std::uniform_int_distribution<int> y_dist( 0, std::max( 0, static_cast<int>( m_rows ) - side ) );
	for ( std::size_t r = 0; r < in.rows(); ++r ) {
		float *const in_row = in.data() + r * flat;
		float *const out_row = out.data() + r * flat;
		cv::Mat img;
		tensor_row_to_mat( in_row, m_rows, m_cols, m_channels, img );
		for ( int p = 0; p < m_num_patches; ++p ) {
			int const x0 = x_dist( m_rng );
			int const y0 = y_dist( m_rng );
			img( cv::Rect( x0, y0, side, side ) ) = 0.0f;
		}
		mat_to_tensor_row_planar( img, out_row, m_rows, m_cols, m_channels );
	}
}

static_assert( cifar10_augment_flat == cifar10_input_dim,
               "CIFAR-10 augmentation flat dim must match cifar10_loader.hpp" );

std::unique_ptr<ImageAugmentation> make_cifar10_training_augmentation()
{
	std::vector<std::unique_ptr<ImageAugmentation>> v;
	v.reserve( 7 );
	v.push_back( std::make_unique<RandomHorizontalFlip>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 0.5f ) );
	v.push_back( std::make_unique<RandomRotation>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 359.0f ) );
	v.push_back( std::make_unique<RandomCrop>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 0.75f ) );
	v.push_back( std::make_unique<ColorJitter>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 0.05f, 0.05f, 0.05f, 8.0f ) );
	v.push_back( std::make_unique<Affine>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 0.0f, 0.12f ) );
	v.push_back( std::make_unique<Noise>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 0.012f ) );
	v.push_back( std::make_unique<RandomErasing>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 0.06f, 0.3f, 3.3f ) );
	return std::make_unique<ImageAugmentationCombinatorial>(
	    cifar10_augment_rows, cifar10_augment_cols, cifar10_augment_ch, 1.0f, std::move( v ) );
}

Cifar10AugmentBatchOp::Cifar10AugmentBatchOp( std::size_t in_cols, std::size_t out_cols,
                                              std::size_t batch_rows,
                                              std::vector<std::unique_ptr<ImageAugmentation>> &&augmentations )
    : input_cols( in_cols )
    , target_cols( out_cols )
    , batch_rows( batch_rows )
    , augmentations( std::move( augmentations ) )
    , tmp_in( omp_get_max_threads(), Tensor<float>( 1, in_cols ) )
    , tmp_out( omp_get_max_threads(), Tensor<float>( 1, in_cols ) )
{
	(void)target_cols;
	if ( in_cols != cifar10_augment_flat ) {
		throw std::invalid_argument(
		    "Cifar10AugmentBatchOp: input_cols must equal cifar10_augment_flat (3072)" );
	}
}

void Cifar10AugmentBatchOp::operator()( Tensor<float> &host_x, Tensor<float> &host_y ) const
{
	(void)host_y;
	if ( augmentations.empty() ) {
		return;
	}
	
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for ( std::size_t r = 0; r < batch_rows; ++r ) {
		std::size_t const thread_id = omp_get_thread_num();
		tmp_in[thread_id].assign( host_x.data() + r * input_cols, input_cols );
		augmentations[thread_id]->apply( tmp_in[thread_id], tmp_out[thread_id] );
		host_x.assignTensorAsRow( r, tmp_out[thread_id] );
	}
}

} // namespace neural
