// Machine-generated; included from nn_io.hpp inside neural::refactor.

template <typename T, typename Device>
inline void apply_layer_meta( LayerBase<T, Device> &layer,
                              std::map<std::string, std::vector<std::byte>> const &m )
{
	using namespace nn_io_detail;
	layer.layerName( ascii_string( m.at( "name" ) ) );
	layer.layerId( parse_ascii_u32( m.at( "id" ) ) );
}

template <typename T, typename Device>
void write( LinearLayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	using namespace nn_io_detail;
	if ( layer.numWeightParams() == 0U ) {
		throw std::runtime_error(
		    "nn_io: LinearLayer has no weights (run forward once before save)" );
	}
	std::size_t const out_f = layer.outFeatures();
	std::size_t const in_f  = layer.numWeightParams() / out_f;
	LinearLayer<T, Device> &mut = const_cast<LinearLayer<T, Device> &>( layer );
	std::vector<T> wflat;
	std::vector<T> bflat;
	if constexpr ( std::is_same_v<Device, Cpu> ) {
		wflat.resize( layer.numWeightParams() );
		bflat.resize( layer.numBiasParams() );
		std::memcpy( wflat.data(), mut.getWeights(),
		             layer.numWeightParams() * sizeof( T ) );
		std::memcpy( bflat.data(), mut.getBias(), layer.numBiasParams() * sizeof( T ) );
	}
#if NEURAL_CUDA_ENABLED
	else {
		cuda_memcpy_d2h( mut.getWeights(), layer.numWeightParams(), wflat );
		cuda_memcpy_d2h( mut.getBias(), layer.numBiasParams(), bflat );
	}
#endif
	write_kv_empty( os, "LAYER_START" );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::Linear ) ) );
	write_kv_ascii( os, "out_features", std::to_string( out_f ) );
	write_kv_ascii( os, "in_features", std::to_string( in_f ) );
	write_tensor_blob_kv( os, "weights", file_dt, wflat.data(), wflat.size() );
	write_tensor_blob_kv( os, "bias", file_dt, bflat.data(), bflat.size() );
	write_kv_empty( os, "LAYER_END" );
}

template <typename T, typename Device>
void write( ConvolutionalLayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	using namespace nn_io_detail;
	if ( layer.numWeightParams() == 0U ) {
		throw std::runtime_error(
		    "nn_io: ConvolutionalLayer has no weights (initialize before save)" );
	}
	std::vector<T> wflat;
	std::vector<T> bflat;
	ConvolutionalLayer<T, Device> &mut = const_cast<ConvolutionalLayer<T, Device> &>( layer );
	if constexpr ( std::is_same_v<Device, Cpu> ) {
		wflat.resize( layer.numWeightParams() );
		bflat.resize( layer.numBiasParams() );
		std::memcpy( wflat.data(), mut.getWeights(),
		             layer.numWeightParams() * sizeof( T ) );
		std::memcpy( bflat.data(), mut.getBias(), layer.numBiasParams() * sizeof( T ) );
	}
#if NEURAL_CUDA_ENABLED
	else {
		cuda_memcpy_d2h( mut.getWeights(), layer.numWeightParams(), wflat );
		cuda_memcpy_d2h( mut.getBias(), layer.numBiasParams(), bflat );
	}
#endif
	write_kv_empty( os, "LAYER_START" );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::Conv ) ) );
	write_kv_ascii( os, "out_channels", std::to_string( mut.outChannels() ) );
	write_kv_ascii( os, "kernel_size", std::to_string( mut.kernelSize() ) );
	write_kv_ascii( os, "pad_h", std::to_string( mut.cudnnPadH() ) );
	write_kv_ascii( os, "pad_w", std::to_string( mut.cudnnPadW() ) );
	if ( auto const ichw = layer.inputChw(); ichw.has_value() ) {
		write_kv_ascii( os, "input_c", std::to_string( ( *ichw )[0] ) );
		write_kv_ascii( os, "input_h", std::to_string( ( *ichw )[1] ) );
		write_kv_ascii( os, "input_w", std::to_string( ( *ichw )[2] ) );
	}
	write_tensor_blob_kv( os, "weights", file_dt, wflat.data(), wflat.size() );
	write_tensor_blob_kv( os, "bias", file_dt, bflat.data(), bflat.size() );
	write_kv_empty( os, "LAYER_END" );
}

template <typename T, typename Device>
void write( BatchNormLayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	using namespace nn_io_detail;
	if ( layer.numWeightParams() == 0U ) {
		throw std::runtime_error( "nn_io: BatchNormLayer not initialized before save" );
	}
	std::size_t const n = layer.numWeightParams();
	BatchNormLayer<T, Device> &mut =
	    const_cast<BatchNormLayer<T, Device> &>( layer );
	std::vector<T> g( n ), bvec( n ), rm( n ), rv( n );
	T const *mean_src = nullptr;
	T const *var_src = nullptr;
	mut.checkpointRunningStatsPtrs( &mean_src, &var_src );
	if ( mean_src == nullptr || var_src == nullptr ) {
		throw std::runtime_error( "nn_io: BatchNormLayer has no checkpointable running stats" );
	}
	if constexpr ( std::is_same_v<Device, Cpu> ) {
		std::memcpy( g.data(), mut.getWeights(), n * sizeof( T ) );
		std::memcpy( bvec.data(), mut.getBias(), n * sizeof( T ) );
		std::memcpy( rm.data(), mean_src, n * sizeof( T ) );
		std::memcpy( rv.data(), var_src, n * sizeof( T ) );
	}
#if NEURAL_CUDA_ENABLED
	else {
		cuda_memcpy_d2h( mut.getWeights(), n, g );
		cuda_memcpy_d2h( mut.getBias(), n, bvec );
		cuda_memcpy_d2h( const_cast<T *>( mean_src ), n, rm );
		cuda_memcpy_d2h( const_cast<T *>( var_src ), n, rv );
	}
#endif
	write_kv_empty( os, "LAYER_START" );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::BatchNorm ) ) );
	write_kv_ascii( os, "num_features", std::to_string( n ) );
	write_double_kv( os, "eps", static_cast<double>( mut.eps() ) );
	write_double_kv( os, "momentum", static_cast<double>( mut.momentum() ) );
	write_tensor_blob_kv( os, "gamma", file_dt, g.data(), g.size() );
	write_tensor_blob_kv( os, "beta", file_dt, bvec.data(), bvec.size() );
	write_tensor_blob_kv( os, "running_mean", file_dt, rm.data(), rm.size() );
	write_tensor_blob_kv( os, "running_var", file_dt, rv.data(), rv.size() );
	write_kv_empty( os, "LAYER_END" );
}

template <typename T, typename Device>
void write( DropoutLayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	(void)file_dt;
	using namespace nn_io_detail;
	write_kv_empty( os, "LAYER_START" );
	DropoutLayer<T, Device> &mut = const_cast<DropoutLayer<T, Device> &>( layer );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::Dropout ) ) );
	write_double_kv( os, "keep_prob", static_cast<double>( mut.keepProb() ) );
	write_kv_ascii( os, "seed", std::to_string( static_cast<unsigned long long>( mut.rngSeed() ) ) );
	write_kv_empty( os, "LAYER_END" );
}

template <typename T, typename Device>
void write( MaxPoolLayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	(void)file_dt;
	using namespace nn_io_detail;
	write_kv_empty( os, "LAYER_START" );
	MaxPoolLayer<T, Device> &mut = const_cast<MaxPoolLayer<T, Device> &>( layer );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::MaxPool ) ) );
	write_kv_ascii( os, "pool_size", std::to_string( mut.poolSize() ) );
	write_kv_ascii( os, "stride", std::to_string( mut.stride() ) );
	write_kv_empty( os, "LAYER_END" );
}

template <typename T, typename Device>
void write( ReLULayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	(void)file_dt;
	using namespace nn_io_detail;
	write_kv_empty( os, "LAYER_START" );
	ReLULayer<T, Device> &mut = const_cast<ReLULayer<T, Device> &>( layer );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::ReLU ) ) );
	write_kv_empty( os, "LAYER_END" );
}

template <typename T, typename Device>
void write( SigmoidLayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	(void)file_dt;
	using namespace nn_io_detail;
	write_kv_empty( os, "LAYER_START" );
	SigmoidLayer<T, Device> &mut = const_cast<SigmoidLayer<T, Device> &>( layer );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::Sigmoid ) ) );
	write_kv_empty( os, "LAYER_END" );
}

template <typename T, typename Device>
void write( SoftmaxLayer<T, Device> const &layer, std::ostream &os, DType file_dt )
{
	(void)file_dt;
	using namespace nn_io_detail;
	write_kv_empty( os, "LAYER_START" );
	SoftmaxLayer<T, Device> &mut = const_cast<SoftmaxLayer<T, Device> &>( layer );
	write_kv_ascii( os, "name", mut.layerName() );
	write_kv_ascii( os, "id", std::to_string( mut.layerId() ) );
	write_kv_ascii( os, "type",
	                std::to_string( static_cast<unsigned>( LayerType::Softmax ) ) );
	write_kv_empty( os, "LAYER_END" );
}

// ---------------------------------------------------------------------------
// read layer maps -> concrete layer
// ---------------------------------------------------------------------------

template <typename T, typename Device>
LinearLayer<T, Device> read_linear_from_map(
    std::map<std::string, std::vector<std::byte>> const &m, DType file_dt )
{
	using namespace nn_io_detail;
	require_key( m, "out_features" );
	require_key( m, "in_features" );
	require_key( m, "weights" );
	require_key( m, "bias" );
	std::size_t const out_f =
	    static_cast<std::size_t>( parse_ascii_u32( m.at( "out_features" ) ) );
	std::size_t const in_f =
	    static_cast<std::size_t>( parse_ascii_u32( m.at( "in_features" ) ) );
	std::size_t const nW = out_f * in_f;
	std::size_t const nB = out_f;

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		T *wp = nullptr;
		T *bp = nullptr;
		wp = static_cast<T *>( ::operator new( nW * sizeof( T ) ) );
		bp = static_cast<T *>( ::operator new( nB * sizeof( T ) ) );
		read_tensor_blob_into<T>( m.at( "weights" ), file_dt, wp, nW );
		read_tensor_blob_into<T>( m.at( "bias" ), file_dt, bp, nB );
		LinearLayer<T, Device> L( wp, in_f, out_f, bp );
		apply_layer_meta( L, m );
		return L;
	}
#if NEURAL_CUDA_ENABLED
	else {
		std::vector<T> hw( nW ), hb( nB );
		read_tensor_blob_into<T>( m.at( "weights" ), file_dt, hw.data(), nW );
		read_tensor_blob_into<T>( m.at( "bias" ), file_dt, hb.data(), nB );
		T *dw = nullptr;
		T *db = nullptr;
		cuda_memcpy_h2d( hw.data(), nW, dw );
		cuda_memcpy_h2d( hb.data(), nB, db );
		LinearLayer<T, Device> L( dw, in_f, out_f, db );
		apply_layer_meta( L, m );
		return L;
	}
#endif
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device> read_conv_from_map(
    std::map<std::string, std::vector<std::byte>> const &m, DType file_dt )
{
	using namespace nn_io_detail;
	require_key( m, "out_channels" );
	require_key( m, "kernel_size" );
	require_key( m, "pad_h" );
	require_key( m, "pad_w" );
	require_key( m, "weights" );
	require_key( m, "bias" );
	std::size_t const out_c =
	    static_cast<std::size_t>( parse_ascii_u32( m.at( "out_channels" ) ) );
	std::size_t const k =
	    static_cast<std::size_t>( parse_ascii_u32( m.at( "kernel_size" ) ) );
	std::int32_t const ph = parse_ascii_i32( m.at( "pad_h" ) );
	std::int32_t const pw = parse_ascii_i32( m.at( "pad_w" ) );
	if ( ph < 0 || pw < 0 ) {
		throw std::runtime_error( "nn_io: conv pad must be non-negative" );
	}
	std::size_t const pad_hu = static_cast<std::size_t>( ph );
	std::size_t const pad_wu = static_cast<std::size_t>( pw );

	std::vector<std::byte> const &wb = m.at( "weights" );
	auto const [inner, wcount64, payload] = nn_io_detail::parse_prefixed_blob_le( wb, file_dt );
	(void)payload;
	(void)inner;
	std::size_t const wcount = static_cast<std::size_t>( wcount64 );
	std::size_t const denom = out_c * k * k;
	if ( denom == 0U || wcount % denom != 0U ) {
		throw std::runtime_error( "nn_io: conv weights count inconsistent" );
	}
	std::size_t const in_c = wcount / denom;
	std::vector<std::size_t> shape{ out_c, in_c, k, k };

	std::optional<std::array<std::size_t, 3>> input_chw;
	if ( m.find( "input_c" ) != m.end() ) {
		require_key( m, "input_h" );
		require_key( m, "input_w" );
		std::size_t const ic =
		    static_cast<std::size_t>( parse_ascii_u32( m.at( "input_c" ) ) );
		std::size_t const ih =
		    static_cast<std::size_t>( parse_ascii_u32( m.at( "input_h" ) ) );
		std::size_t const iw =
		    static_cast<std::size_t>( parse_ascii_u32( m.at( "input_w" ) ) );
		if ( ic != in_c ) {
			throw std::runtime_error(
			    "nn_io: input_c does not match conv weight tensor in_channels" );
		}
		input_chw = std::array<std::size_t, 3>{ ic, ih, iw };
	}

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		T *wp = static_cast<T *>( ::operator new( wcount * sizeof( T ) ) );
		read_tensor_blob_into<T>( m.at( "weights" ), file_dt, wp, wcount );
		std::vector<T> hbvec( out_c );
		read_tensor_blob_into<T>( m.at( "bias" ), file_dt, hbvec.data(), out_c );
		T *bp = static_cast<T *>( ::operator new( out_c * sizeof( T ) ) );
		std::memcpy( bp, hbvec.data(), out_c * sizeof( T ) );
		ConvolutionalLayer<T, Device> L( wp, shape, bp, pad_hu, pad_wu, input_chw );
		apply_layer_meta( L, m );
		return L;
	}
#if NEURAL_CUDA_ENABLED
	else {
		std::vector<T> hw( wcount ), hbvec( out_c );
		read_tensor_blob_into<T>( m.at( "weights" ), file_dt, hw.data(), wcount );
		read_tensor_blob_into<T>( m.at( "bias" ), file_dt, hbvec.data(), out_c );
		T *dw = nullptr;
		T *db = nullptr;
		cuda_memcpy_h2d( hw.data(), wcount, dw );
		cuda_memcpy_h2d( hbvec.data(), out_c, db );
		ConvolutionalLayer<T, Device> L( dw, shape, db, pad_hu, pad_wu, input_chw );
		apply_layer_meta( L, m );
		return L;
	}
#endif
}

template <typename T, typename Device>
BatchNormLayer<T, Device> read_batchnorm_from_map(
    std::map<std::string, std::vector<std::byte>> const &m, DType file_dt )
{
	using namespace nn_io_detail;
	require_key( m, "num_features" );
	std::size_t const C =
	    static_cast<std::size_t>( parse_ascii_u32( m.at( "num_features" ) ) );
	require_key( m, "eps" );
	require_key( m, "momentum" );
	T const eps_val = static_cast<T>( read_double_bin( m.at( "eps" ) ) );
	T const mom_val = static_cast<T>( read_double_bin( m.at( "momentum" ) ) );
	require_key( m, "gamma" );
	require_key( m, "beta" );
	require_key( m, "running_mean" );
	require_key( m, "running_var" );

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		T *g       = static_cast<T *>( ::operator new( C * sizeof( T ) ) );
		T *bptr    = static_cast<T *>( ::operator new( C * sizeof( T ) ) );
		T *meanptr = static_cast<T *>( ::operator new( C * sizeof( T ) ) );
		T *varptr  = static_cast<T *>( ::operator new( C * sizeof( T ) ) );
		read_tensor_blob_into<T>( m.at( "gamma" ), file_dt, g, C );
		read_tensor_blob_into<T>( m.at( "beta" ), file_dt, bptr, C );
		read_tensor_blob_into<T>( m.at( "running_mean" ), file_dt, meanptr, C );
		read_tensor_blob_into<T>( m.at( "running_var" ), file_dt, varptr, C );
		BatchNormLayer<T, Device> L( g, bptr, meanptr, varptr, C, eps_val, mom_val );
		apply_layer_meta( L, m );
		return L;
	}
#if NEURAL_CUDA_ENABLED
	else {
		std::vector<T> hg( C ), hb( C ), hm( C ), hv( C );
		read_tensor_blob_into<T>( m.at( "gamma" ), file_dt, hg.data(), C );
		read_tensor_blob_into<T>( m.at( "beta" ), file_dt, hb.data(), C );
		read_tensor_blob_into<T>( m.at( "running_mean" ), file_dt, hm.data(), C );
		read_tensor_blob_into<T>( m.at( "running_var" ), file_dt, hv.data(), C );
		T *dg = nullptr;
		T *db = nullptr;
		T *dm = nullptr;
		T *dv = nullptr;
		cuda_memcpy_h2d( hg.data(), C, dg );
		cuda_memcpy_h2d( hb.data(), C, db );
		cuda_memcpy_h2d( hm.data(), C, dm );
		cuda_memcpy_h2d( hv.data(), C, dv );
		BatchNormLayer<T, Device> L( dg, db, dm, dv, C, eps_val, mom_val );
		apply_layer_meta( L, m );
		return L;
	}
#endif
}

template <typename T, typename Device>
DropoutLayer<T, Device> read_dropout_from_map(
    std::map<std::string, std::vector<std::byte>> const &m, DType file_dt )
{
	(void)file_dt;
	using namespace nn_io_detail;
	require_key( m, "keep_prob" );
	require_key( m, "seed" );
	T const kp = static_cast<T>( read_double_bin( m.at( "keep_prob" ) ) );
	std::uint64_t const seed64 = parse_ascii_u64( m.at( "seed" ) );
	std::uint32_t const seed32 = static_cast<std::uint32_t>( seed64 & 0xFFFFFFFFULL );
	DropoutLayer<T, Device> L( kp, seed32 );
	apply_layer_meta( L, m );
	return L;
}

template <typename T, typename Device>
MaxPoolLayer<T, Device> read_maxpool_from_map(
    std::map<std::string, std::vector<std::byte>> const &m, DType file_dt )
{
	(void)file_dt;
	using namespace nn_io_detail;
	require_key( m, "pool_size" );
	require_key( m, "stride" );
	std::size_t const ps = static_cast<std::size_t>( parse_ascii_u32( m.at( "pool_size" ) ) );
	std::size_t const st = static_cast<std::size_t>( parse_ascii_u32( m.at( "stride" ) ) );
	MaxPoolLayer<T, Device> L( ps, st );
	apply_layer_meta( L, m );
	return L;
}

template <typename T, typename Device>
Layer<T, Device> read_layer_from_map( std::map<std::string, std::vector<std::byte>> const &m,
                                      DType file_dt )
{
	using namespace nn_io_detail;
	require_key( m, "type" );
	LayerType const lt = static_cast<LayerType>( parse_ascii_u32( m.at( "type" ) ) );
	switch ( lt ) {
	case LayerType::Linear:
		return read_linear_from_map<T, Device>( m, file_dt );
	case LayerType::Conv:
		return read_conv_from_map<T, Device>( m, file_dt );
	case LayerType::BatchNorm:
		return read_batchnorm_from_map<T, Device>( m, file_dt );
	case LayerType::Dropout:
		return read_dropout_from_map<T, Device>( m, file_dt );
	case LayerType::MaxPool:
		return read_maxpool_from_map<T, Device>( m, file_dt );
	case LayerType::ReLU: {
		ReLULayer<T, Device> L;
		apply_layer_meta( L, m );
		return L;
	}
	case LayerType::Sigmoid: {
		SigmoidLayer<T, Device> L;
		apply_layer_meta( L, m );
		return L;
	}
	case LayerType::Softmax: {
		SoftmaxLayer<T, Device> L;
		apply_layer_meta( L, m );
		return L;
	}
	default:
		throw std::runtime_error( "nn_io: unknown LayerType" );
	}
}

template <typename T, typename Device>
void save( SequentialNN2<T, Device> &net, std::string const &path )
{
	std::ofstream os( path, std::ios::binary );
	if ( !os ) {
		throw std::runtime_error( "nn_io: cannot open file for write: " + path );
	}
	Header h{};
	h.magic[0]   = 'N';
	h.magic[1]   = 'N';
	h.magic[2]   = 'C';
	h.magic[3]   = 'X';
	h.version    = kNnFormatVersion;
	h.dtype      = dtype_of<T>;
	h.num_layers = static_cast<std::uint32_t>( net.numLayers() );
	h.nn_type    = 0;
	nn_io_detail::write_file_header( os, h );
	net.forEachLayer( [&]( auto const &lyr ) { write( lyr, os, h.dtype ); } );
}

template <typename T, typename Device>
void load( SequentialNN2<T, Device> &net, std::string const &path )
{
	std::ifstream is( path, std::ios::binary );
	if ( !is ) {
		throw std::runtime_error( "nn_io: cannot open file for read: " + path );
	}
	Header const h = nn_io_detail::read_file_header( is );
	net.clear();
	for ( std::uint32_t i = 0; i < h.num_layers; ++i ) {
		std::map<std::string, std::vector<std::byte>> m;
		nn_io_detail::read_layer_map( is, m );
		net.addLayer( read_layer_from_map<T, Device>( m, h.dtype ) );
	}
	net.wire();
}
