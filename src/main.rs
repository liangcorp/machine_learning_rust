use core::panic;
use std::env;
pub use std::path::Path;

use ml_rust::linear_regression::cost_functions;
use ml_rust::linear_regression::gradient_descent;
use ml_rust::linear_regression::normal_equation;
use ml_rust::read_data;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Error: no input argument");
        std::process::exit(exitcode::DATAERR);
    }

    if args[1].is_empty() {
        eprintln!("Error: filename is empty");
        std::process::exit(exitcode::DATAERR);
    }

    let file_path = Path::new(&args[1]);

    let (x_ptr, y_ptr) = match read_data::get_data(file_path) {
        Ok((x_ptr, y_ptr)) => (x_ptr, y_ptr),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    };

    // testing linear regression
    let x = *x_ptr;
    let y = *y_ptr;

    let alpha = 0.01; // the learning speed
    let num_iters = 2000; // Number of gradient descent iterations
    let mut theta = vec![0.0, 0.0]; // set theta 0 and theta 1 to 0.0

    match cost_functions::get_cost(&x, &y, &theta) {
        Ok(theta) => {
            println!("Thetas are [0.0, 0.0], J(theta) is {:?}", theta);
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match cost_functions::get_cost(&x, &y, &[-1.0, 2.0]) {
        Ok(theta) => {
            println!("Thetas are [-1.0, 2.0], J(theta) is {:?}", theta);
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match gradient_descent::get_thetas(&x, &y, alpha, &mut theta, num_iters) {
        Ok(theta) => println!("Found thetas using Gradient Descent: {:?}", theta),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match normal_equation::get_theta(&x, &y) {
        Ok(theta) => {
            println!("Found thetas using Normal Equation: {:?}", theta)
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }
}
