/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use std::{path::PathBuf, process::Command};

fn main() {
    let mut out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_dir.push("lib");

    if !out_dir.is_dir() {
        std::fs::create_dir(&out_dir).unwrap();
    }

    let add_obj_file = out_dir.join("add.o");
    let sub_obj_file = out_dir.join("sub.o");
    let lib_file = out_dir.join("libtest_wasm32.a");

    let output = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/build_test_lib.py"
    ))
    .arg(&out_dir)
    .output()
    .expect("Failed to execute command");
    assert!(
        add_obj_file.exists(),
        "Could not build tvm lib: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    let ar = "llvm-ar-9";
    let output = Command::new(ar)
        .arg("rcs")
        .arg(&lib_file)
        .arg(&add_obj_file)
        .arg(&sub_obj_file)
        .output()
        .expect("Failed to execute command");
    assert!(
        lib_file.exists(),
        "Could not create archive: {}",
        String::from_utf8(output.stderr)
            .unwrap()
            .trim()
            .split("\n")
            .last()
            .unwrap_or("")
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
