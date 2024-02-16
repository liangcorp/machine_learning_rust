//! Implementation of linear regression
use crate::read_data;
use std::path::Path;

pub mod cost_functions;
pub mod gradient_descent;
pub mod normal_equation;

// Sample run of linear regression
pub fn sample_run(input_file_path: &Path) {
    let alpha = 0.1; // the learning speed
    let num_iters = 5000; // Number of gradient descent iterations

    // Read data from file
    let (x, y) = match read_data::get_data(input_file_path) {
        Ok((x_ptr, y_ptr)) => (*x_ptr, *y_ptr),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    };
    let theta = vec![0.0; x[0].len()]; // set theta 0 and theta 1 to 0.0

    match gradient_descent::get_thetas(&x, &y, alpha, &theta, num_iters) {
        Ok(theta) => {
            print!("Found thetas using Gradient Descent with learning speed {} and {} number of iterations (skipping theta 0): [", alpha, num_iters);
            for t in theta.iter().skip(1) {
                print!(" {} ", t);
            }
            println!("]");
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match normal_equation::get_theta(&x, &y) {
        Ok(theta) => {
            print!("Found thetas using Normal Equation (skipping theta 0): [");
            for t in theta.iter().skip(1) {
                print!(" {} ", t);
            }
            println!("]");
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match cost_functions::get_cost(&x, &y, &theta) {
        Ok(j_theta) => {
            println!("Thetas are {:?}, J(theta) is {:?}", theta, j_theta);
        }
        Err(e) => eprint!("{}", e.get_ref().unwrap()),
    }
}
