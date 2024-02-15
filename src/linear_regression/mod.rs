//! Implementation of linear regression
pub mod cost_functions;
pub mod gradient_descent;
pub mod normal_equation;

// Sample run of linear regression
pub fn sample_run(x: &[Vec<f64>], y: &[f64]) {
    let alpha = 0.01; // the learning speed
    let num_iters = 2500; // Number of gradient descent iterations
    let mut theta = vec![0.0; x[0].len()]; // set theta 0 and theta 1 to 0.0

    match gradient_descent::get_thetas(x, y, alpha, &mut theta, num_iters) {
        Ok(theta) => {
            print!("Found thetas using Gradient Descent (skipping theta 0): [");
            for t in theta.iter().skip(1) {
                print!(" {} ", t);
            }
            println!("]");
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match normal_equation::get_theta(x, y) {
        Ok(theta) => {
            print!("Found thetas using Normal Equation (skipping theta 0): [");
            for t in theta.iter().skip(1) {
                print!(" {} ", t);
            }
            println!("]");
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    match cost_functions::get_cost(x, y, &theta) {
        Ok(j_theta) => {
            println!("Thetas are {:?}, J(theta) is {:?}", theta, j_theta);
        }
        Err(e) => eprint!("{}", e.get_ref().unwrap()),
    }
}
