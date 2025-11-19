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

        if(!is_valid())
        {
            return;
        }

        // This is the core of the new logic
        // Check the number of shared_ptr that co-own's the data
        if (data_ptr_.use_count() > 1)
        {
            throw std::runtime_error("Buffer is used by other Tensors or views and cannot be released at the moment");
        }

        this->data_ptr_.reset();



        if (this->grad_ptr_)
        {
            // If the count is grater than 1, it means another tensor (a view or a copy)
            // is still using this memory. Throwing an exception to prevent corruption
            if (grad_ptr_.use_count())
            {
                throw std::runtime_error("Gradient Buffer is used by other tensors or views and cannot be released at the moment");
            }
        this->grad_ptr_.reset();
        }  
    }

}