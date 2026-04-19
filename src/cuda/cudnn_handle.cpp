#include "cudnn_handle.hpp"
#include <cudnn.h>
#include <stdexcept>

using namespace neural;

cudnnHandle_t CudnnHandle::m_cudnn_handle = nullptr;

CudnnHandle::CudnnHandle()
{
	if ( !m_cudnn_handle ) {
		cudnnStatus_t const st = cudnnCreate( &m_cudnn_handle );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnCreate failed" );
		}
	} else {
		throw std::runtime_error( "CudnnHandle already initialized" );
	}
}

CudnnHandle::~CudnnHandle()
{
	if ( m_cudnn_handle ) {
		cudnnDestroy( m_cudnn_handle );
		m_cudnn_handle = nullptr;
	}
}

cudnnHandle_t &CudnnHandle::instance()
{
	if ( !m_cudnn_handle ) {
		cudnnStatus_t const st = cudnnCreate( &m_cudnn_handle );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnCreate failed" );
		}
	}
	return m_cudnn_handle;
}
