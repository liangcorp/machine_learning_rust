//! Cost function with multiple features
//!
//! This crate perform calculation of J(theta)

use std::io;
use std::io::{Error, ErrorKind};
/// # Cost function for multiple features (x\[1\], x\[2\], ..., x\[n\]
///
/// - X and y are the training sets.
/// - X is a 2D Vector represent multiple training sets
/// - theta is a vector that contains chosen numbers.
///
/// ## Implement the following matlab formula:
///
/// hx = 1 ./ (1 + exp(-(theta' * X')));
///
/// J = sum(-y .* log(hx') - (1 - y) .* log(1 - hx')) /
///         m + (lambda / (2 * m)) * sum(theta_without_first.^2);
///

pub fn get_cost(x: &[Vec<64>], y: &[f64], theta: &[f64]) -> Result<f64, io::Error> {
    let num_feat = theta.len();
    let mut h_x: Vec<f64> = vec![];

    let mut j_theta: f64 = 0.0;     // The cost
    let mut sum: f64;

    let num_train = if x.len() == y.len() {
        y.len()
    } else {
        return Err(Error::new(ErrorKind::Other, "Mis-match training sets"));
    };

    Ok(j_theta)
}
