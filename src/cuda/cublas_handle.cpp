#include "cublas_handle.hpp"
#include <cublas_v2.h>
#include <stdexcept>

using namespace neural;

cublasHandle_t CublasHandle::m_cublas_handle = nullptr;

CublasHandle::CublasHandle()
{
	if( !m_cublas_handle ) {
		cublasCreate( &m_cublas_handle );
	} else {
		throw std::runtime_error( "CublasHandle already initialized" );
	}
}
CublasHandle::~CublasHandle()
{
	if( m_cublas_handle ) {
		cublasDestroy( m_cublas_handle );
		m_cublas_handle = nullptr;
	}
}

cublasHandle_t &CublasHandle::instance()
{
	if ( !m_cublas_handle ) {
		cublasStatus_t const st = cublasCreate( &m_cublas_handle );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "cublasCreate failed" );
		}
	}
	return m_cublas_handle;
}