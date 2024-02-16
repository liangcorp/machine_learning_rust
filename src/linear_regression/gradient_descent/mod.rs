//! Gradient descent with multiple features
//!
//! This crate is a collection of functions to perform
//! calculation of gradient descent

use std::io;
use std::io::{Error, ErrorKind};

/// # Gradient descent for a single feature (x\[1\])
///
/// - X and y are the training sets.
/// - alpha is the learning rate
/// - theta is a chosen number.
///
/// ## Implement the following matlab formula:
///
///
/// theta(indx,:) = theta(indx,:) -
///                 alpha * ((((temp[] * X[]) - y[]) * X(:,indx))/m);
///
///
pub fn get_thetas(
    x_mtrx: &[Vec<f64>],
    y_vec: &[f64],
    alpha: f64,
    theta: &mut [f64],
    iterations: u32,
) -> Result<Box<Vec<f64>>, io::Error> {
    let m = y_vec.len(); // no of training sets
    let num_feat = theta.len();

    let mut sum: f64;
    let mut tmp_theta: Vec<f64>;
    let mut h_x: Vec<f64> = Vec::new();

    if x_mtrx.len() != y_vec.len() {
        return Err(Error::new(ErrorKind::Other, "Mis-matching training sets"));
    }

    for _ in 0..iterations {
        h_x.clear();

        tmp_theta = theta.to_owned();

        for x_row in x_mtrx.iter() {
            sum = 0.0;

            for j in 0..num_feat {
                sum += tmp_theta[j] * x_row[j];
            }

            h_x.push(sum);
        }

        for j in 0..num_feat {
            sum = 0.0;

            for i in 0..m {
                sum += (h_x[i] - y_vec[i]) * x_mtrx[i][j];
            }

            theta[j] = tmp_theta[j] - (alpha * sum / m as f64);
        }
    }

    Ok(Box::new(theta.to_vec()))
}
