#include "refactor/latch_base.hpp"
#include "refactor/latch_layer.hpp"
#include "refactor/layer_base.hpp"
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cstddef>

using neural::refactor::Cpu;
using neural::refactor::LatchBase;
using neural::refactor::LatchLayer;
using neural::refactor::LayerBase;

TEST_CASE( "LatchLayer<float,Cpu> resize allocates and reports shape", "[latch][cpu][resize]" )
{
	LatchLayer<float, Cpu> latch;
	latch.resize( {4, 8} );

	REQUIRE( latch.shape() == std::vector<std::size_t>{4, 8} );
	REQUIRE( latch.fwdData() != nullptr );
	REQUIRE( latch.bwdData() != nullptr );
}

TEST_CASE( "LatchLayer<float,Cpu> fwd and bwd buffers are independent", "[latch][cpu][buffers]" )
{
	LatchLayer<float, Cpu> latch;
	latch.resize( {3, 4} );

	REQUIRE( latch.fwdData() != latch.bwdData() );
}

TEST_CASE( "LatchLayer<float,Cpu> resize updates size", "[latch][cpu][resize]" )
{
	LatchLayer<float, Cpu> latch;
	latch.resize( {2, 5} );
	REQUIRE( latch.shape()[0] == 2 );
	REQUIRE( latch.shape()[1] == 5 );

	latch.resize( {7, 3} );
	REQUIRE( latch.shape()[0] == 7 );
	REQUIRE( latch.shape()[1] == 3 );
}

TEST_CASE( "LatchBase<float,Cpu>* polymorphism via base pointer", "[latch][cpu][polymorphism]" )
{
	LatchLayer<float, Cpu> latch;
	LatchBase<float, Cpu>* p = &latch;

	p->resize( {2, 16} );

	REQUIRE( p->shape()[0] == 2 );
	REQUIRE( p->shape()[1] == 16 );
	REQUIRE( p->fwdData() != nullptr );
	REQUIRE( p->bwdData() != nullptr );
}

TEST_CASE( "LatchLayer<size_t,Cpu> works for argmax-style buffers", "[latch][cpu][sizet]" )
{
	LatchLayer<std::size_t, Cpu> latch;
	latch.resize( {2, 4, 6, 6} );

	REQUIRE( latch.shape().size() == 4 );
	REQUIRE( latch.fwdData() != nullptr );
}

TEST_CASE( "LayerBase<float,Cpu> wiring — getInput and getOutput return set latches", "[layer_base][cpu][wiring]" )
{
	LatchLayer<float, Cpu> in_latch;
	LatchLayer<float, Cpu> out_latch;
	LayerBase<float, Cpu>  lb;

	lb.setInputOutputLatches( &in_latch, &out_latch );

	REQUIRE( lb.getInput()  == &in_latch );
	REQUIRE( lb.getOutput() == &out_latch );
}

TEST_CASE( "LayerBase<float,Cpu> wiring — getGradInput and getGradOutput return set latches", "[layer_base][cpu][grad_wiring]" )
{
	LatchLayer<float, Cpu> grad_in;
	LatchLayer<float, Cpu> grad_out;
	LayerBase<float, Cpu>  lb;

	lb.setGradLatches( &grad_in, &grad_out );

	REQUIRE( lb.getGradInput()  == &grad_in );
	REQUIRE( lb.getGradOutput() == &grad_out );
}

TEST_CASE( "LayerBase<float,Cpu> default latches are null", "[layer_base][cpu][default]" )
{
	LayerBase<float, Cpu> lb;

	REQUIRE( lb.getInput()      == nullptr );
	REQUIRE( lb.getOutput()     == nullptr );
	REQUIRE( lb.getGradInput()  == nullptr );
	REQUIRE( lb.getGradOutput() == nullptr );
}
