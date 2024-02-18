//! Implementation of linear regression
use crate::read_data;
use std::path::Path;

pub mod cost_functions;
pub mod gradient_descent;
pub mod normal_equation;

const ITERATIONS: u32 = 5000; // the learning speed

// Sample run of linear regression
pub fn sample_run(input_file_path: &Path) {
    // Read data from file
    let (x, flattened_x, y) = match read_data::get_data(input_file_path) {
        Ok((x_ptr, flattened_x_ptr, y_ptr)) => (*x_ptr, *flattened_x_ptr, *y_ptr),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    };
    let mut theta = vec![0.0; x[0].len()]; // set theta 0 and theta 1 to 0.0

    // set the learning rate = no of features / 10
    let alpha = if x[0].len() < 3 {
        0.01
    } else {
        x[0].len() as f32 / 10.0
    };

    // match gradient_descent::get_thetas(&x, &y, alpha, &mut theta, ITERATIONS) {
    //     Ok(theta) => {
    //         print!("Found thetas using Gradient Descent with learning speed {} and {} number of iterations: {:?}", alpha, ITERATIONS, &theta[1..]);
    //     }
    //     Err(e) => panic!("{}", e.get_ref().unwrap()),
    // }

    match gradient_descent::get_thetas_flatten_x(&flattened_x, &y, alpha, &mut theta, ITERATIONS) {
        Ok(theta) => {
            println!("Gradient Descent with learning speed {} and {} number of iterations:\n {:?}", alpha, ITERATIONS, &theta[1..]);
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }
    // match normal_equation::get_theta(&x, &y) {
    //     Ok(theta) => {
    //         print!("Found thetas using Normal Equation (skipping theta 0): [");
    //         for t in theta.iter().skip(1) {
    //             print!(" {} ", t);
    //         }
    //         println!("]");
    //     }
    //     Err(e) => panic!("{}", e.get_ref().unwrap()),
    // }
    //
    // match cost_functions::get_cost(&x, &y, &theta) {
    //     Ok(j_theta) => {
    //         println!("Thetas are {:?}, J(theta) is {:?}", theta, j_theta);
    //     }
    //     Err(e) => eprint!("{}", e.get_ref().unwrap()),
    // }
}
