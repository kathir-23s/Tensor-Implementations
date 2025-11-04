#include "core/Tensor.h"

namespace OwnTensor 
{
    /** 
    * @brief Checks if the tensor has a valid, non null data pointer.
    * To check whether if a tensor is released or not
    */
    bool Tensor::is_valid() const
    {
        return this->data_ptr_ != nullptr;
    }


    /**
    * @brief Releases underlying memory buffers for gradients and data
    * 
    * This function explicitly releases ownership of the memory. The memory is
    * only deallocated if this was the last Tensor object sharing it. After
    * this call, the Tensor is invalid and is_valid() will return false.
    */

    void Tensor::release()
    {
        this->data_ptr_.reset();

        if (this->grad_ptr_)
        {
            this->grad_ptr_.reset();
        }
    }

}