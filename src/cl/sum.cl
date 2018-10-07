// TODO


#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE   256


__kernel void sum(__global const int *values, unsigned int size, __global unsigned int *res) {
   int localId = get_local_id(0);
   int globalId = get_global_id(0);

   __local unsigned int locals[WORK_GROUP_SIZE];

   if (globalId < size) {

   locals[localId] = values[globalId];

   barrier(CLK_LOCAL_MEM_FENCE);

   for (int nvalues = WORK_GROUP_SIZE; 	nvalues > 1; nvalues /= 2) {
       if (2 * localId < nvalues) {
           int a = locals[localId];
           int b = locals[localId + nvalues/2];

           locals[localId] = a + b;

       }
       barrier(CLK_LOCAL_MEM_FENCE);
   }
   if (localId == 0) {
       atomic_add(res, locals[0]);
   }
 }
}

