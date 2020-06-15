extern "C" {
    static __tvm_module_ctx: i32;
}

#[no_mangle]
unsafe fn __get_tvm_module_ctx() -> i32 {
    // Refer a symbol in the libtest_wasm32.a to make sure that the link of the
    // library is not optimized out.
    __tvm_module_ctx
}

extern crate ndarray;
#[macro_use]
extern crate tvm_runtime;

use ndarray::Array;
use tvm_runtime::{DLTensor, Module as _, SystemLibModule};

#[no_mangle]
pub extern "C" fn run() {
    // try static
    let mut a = Array::from_vec(vec![1f32, 2., 3., 4.]);
    let mut b = Array::from_vec(vec![1f32, 0., 1., 0.]);
    let mut c = Array::from_vec(vec![0f32; 4]);
}
