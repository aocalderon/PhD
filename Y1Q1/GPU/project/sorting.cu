#include <thrust/device_vector.h>
#include <thrust/sort.h>

struct MyStruct1
{
   int key;
   int value1;
   int value2;
};

struct MyStruct2
{
   int N;
   int* key;
   int* value1;
   int* value2;

   MyStruct2(int N_) {
      N = N_;
      cudaMalloc((void**)&key,N*sizeof(int));
      cudaMalloc((void**)&value1,N*sizeof(int));
      cudaMalloc((void**)&value2,N*sizeof(int));
   }
};

__host__ __device__ bool operator<(const MyStruct1 &lhs, const MyStruct1 &rhs) { return (lhs.key < rhs.key); };

int main(void)
{
   const int N = 100000;

   float time;
   cudaEvent_t start, stop;

   /*******************************/
   /* SORTING ARRAY OF STRUCTURES */
   /*******************************/
   thrust::host_vector<MyStruct1> h_struct1(N);
   for (int i = 0; i<N; i++)
   {
      MyStruct1 s;
      s.key        = rand()*255;
      s.value1    = rand()*255;
      s.value2    = rand()*255;
      h_struct1[i]    = s;
   }
   thrust::device_vector<MyStruct1> d_struct(h_struct1);

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);

   thrust::sort(d_struct.begin(), d_struct.end());

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   printf("Sorting array of structs - elapsed time:  %3.1f ms \n", time);

   h_struct1 = d_struct;

   //for (int i = 0; i<N; i++)
   //{
   //    MyStruct1 s = h_struct1[i];
   //    printf("key %i value1 %i value2 %i\n",s.key,s.value1,s.value2);
   //}
   //printf("\n\n");

   /*******************************/
   /* SORTING STRUCTURES OF ARRAYS*/
   /*******************************/

   MyStruct2 d_struct2(N);
   thrust::host_vector<int> h_temp_key(N);
   thrust::host_vector<int> h_temp_value1(N);
   thrust::host_vector<int> h_temp_value2(N);

   //for (int i = 0; i<N; i++)
   //{
   //    h_temp_key[i]        = rand()*255;
   //    h_temp_value1[i]    = rand()*255;
   //    h_temp_value2[i]    = rand()*255;
   //    printf("Original data - key %i value1 %i value2    %i\n",h_temp_key[i],h_temp_value1[i],h_temp_value2[i]);
   //}
   //printf("\n\n");

   cudaMemcpy(d_struct2.key,h_temp_key.data(),N*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(d_struct2.value1,h_temp_value1.data(),N*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(d_struct2.value2,h_temp_value2.data(),N*sizeof(int),cudaMemcpyHostToDevice);

   // wrap raw pointers with device pointers
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);

   thrust::device_ptr<int> dev_ptr_key        =     thrust::device_pointer_cast(d_struct2.key);
   thrust::device_ptr<int> dev_ptr_value1    =     thrust::device_pointer_cast(d_struct2.value1);
   thrust::device_ptr<int> dev_ptr_value2    =     thrust::device_pointer_cast(d_struct2.value2);

   thrust::device_vector<int> d_indices(N);
   thrust::sequence(d_indices.begin(), d_indices.end(), 0, 1);

   // first sort the keys and indices by the keys
   thrust::sort_by_key(dev_ptr_key, dev_ptr_key + N, d_indices.begin());

   // Now reorder the ID arrays using the sorted indices
   thrust::gather(d_indices.begin(), d_indices.end(), dev_ptr_value1, dev_ptr_value1);
   thrust::gather(d_indices.begin(), d_indices.end(), dev_ptr_value2, dev_ptr_value2);

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   printf("Sorting struct of arrays - elapsed time:  %3.1f ms \n", time);

   cudaMemcpy(h_temp_key.data(),d_struct2.key,N*sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(h_temp_value1.data(),d_struct2.value1,N*sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(h_temp_value2.data(),d_struct2.value2,N*sizeof(int),cudaMemcpyDeviceToHost);

   //for (int i = 0; i<N; i++) printf("Ordered data - key %i value1 %i value2 %i\n",h_temp_key[i],h_temp_value1[i],h_temp_value2[i]);
   //printf("\n\n");

   getchar();
   return 0;

}
