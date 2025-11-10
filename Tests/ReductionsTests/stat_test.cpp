#include "TensorLib.h"

using namespace OwnTensor;

int main()
{
    TensorOptions opts;
    opts.dtype = Dtype::Float32;
    opts.device = Device::CPU;
    opts.requires_grad = true;

    Tensor A(Shape{{3,3}}, Dtype::Int32, DeviceIndex(Device::CPU));
    //std::vector<int> dataa= {1,2,3,4,5,6};
    //A.set_data(dataa);
    A.fill(12132);
A.display(std::cout, 5);
    Tensor B = reduce_var(A, {1}, false, 0);
    std::cout << "B:" <<std::endl;
    B.display(std::cout, 5);
//     B = reduce_nanvar(A, {1}, false, 0);
//     std::cout << "B:nanvar" <<std::endl;
//     B.display(std::cout, 5);
    B = reduce_std(A, {1}, false, 0);
    std::cout << "B:" <<std::endl;
    B.display(std::cout, 5);
//      B = reduce_nanstd(A, {1}, false, 0);
//     std::cout << "B:nanstd" <<std::endl;
//     B.display(std::cout, 5);
    std::pair<Tensor, Tensor> C = reduce_var_mean(A, {1}, false, 0);
    std::cout << "C:" <<std::endl;
   for(Tensor t : {C.first, C.second}) 
   {
        t.display(std::cout, 5);
        std::cout << "----" << std::endl;
   }
    C = reduce_std_mean(A, {1}, false, 0);
    std::cout << "C:" <<std::endl;
   for(Tensor t : {C.first, C.second}) 
   {
        t.display(std::cout, 5);
        std::cout << "----" << std::endl;
   }

   std::cout<<"---- CUDA TESTS ----"<<std::endl;

   //TensorOptions opts;
    opts.dtype = Dtype::Float32;
    opts.device = Device::CUDA;
    opts.requires_grad = true;

    Tensor A1(Shape{{3,3}}, Dtype::Int32, DeviceIndex(Device::CUDA));
    //std::vector<int> data = {1,2,3,4,5,6};
    //A1.set_data(data);
    A1.fill(12132);

    B = reduce_var(A1, {1}, false, 0);
    std::cout << "B:" <<std::endl;
    B.to_cpu().display(std::cout, 5);
    B = reduce_nanvar(A1, {1}, false, 0);
    std::cout << "B:nanvar" <<std::endl;
    B.to_cpu().display(std::cout, 5);
    B = reduce_std(A1, {1}, false, 0);
    std::cout << "B: " <<std::endl;
    B.to_cpu().display(std::cout, 5);
     B = reduce_nanstd(A1, {1}, false, 0);
    std::cout << "B: reduce nanstd" <<std::endl;
    B.to_cpu().display(std::cout, 5);
     C = reduce_var_mean(A1, {1}, false, 0);
    std::cout << "C:" <<std::endl;
   for(Tensor t : {C.first, C.second}) 
   {
        t.to_cpu().display(std::cout, 5);
        std::cout << "----" << std::endl;
   }
    C = reduce_std_mean(A1, {1}, false, 0);
    std::cout << "C:" <<std::endl;
   for(Tensor t : {C.first, C.second}) 
   {
        t.to_cpu().display(std::cout, 5);
        std::cout << "----" << std::endl;
   }



   }
    // C.display(std::cout, 5);
    // std::cout << "A:" <<std::endl;
    // A.display(std::cout, 5);

    //Tensor B({{10,10}},opts);
  
//    // B.display(std::cout, 5);
//    Tensor B = reduce_max(A, {1}, true);
//     // Tensor C = B.to_cpu();
//     std::cout << "B:" <<std::endl;
//      B.display(std::cout, 5);

    // std::cout << "A:" <<std::endl;
    // A.display(std::cout, 5);

    // Tensor C = B * A;
    // std::cout << "C:" <<std::endl;
    // C.display(std::cout, 5);
