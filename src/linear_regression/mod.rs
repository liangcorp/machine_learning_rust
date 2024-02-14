//! Implementation of linear regression
pub mod cost_functions;
pub mod gradient_descent;
pub mod normal_equation;

// Sample run of linear regression
pub fn sample_run(x: &[Vec<f64>], y: &[f64]) {
    let alpha = 0.01; // the learning speed
    let num_iters = 2000; // Number of gradient descent iterations
    let mut theta = vec![0.0, 0.0]; // set theta 0 and theta 1 to 0.0

    match cost_functions::get_cost(x, y, &theta) {
        Ok(theta) => {
            println!("Thetas are [0.0, 0.0], J(theta) is {:?}", theta);
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match cost_functions::get_cost(x, y, &[-1.0, 2.0]) {
        Ok(theta) => {
            println!("Thetas are [-1.0, 2.0], J(theta) is {:?}", theta);
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match gradient_descent::get_thetas(x, y, alpha, &mut theta, num_iters) {
        Ok(theta) => println!("Found thetas using Gradient Descent: {:?}", theta),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match normal_equation::get_theta(x, y) {
        Ok(theta) => {
            println!("Found thetas using Normal Equation: {:?}", theta)
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }
}
