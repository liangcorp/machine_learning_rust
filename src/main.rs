use core::panic;
pub use std::path::Path;
use std::{env, io};

use ml_rust::linear_regression;
use ml_rust::read_data;

const ARGS_SIZE_LIMIT: usize = 2;

fn argument_check(args: &[String]) -> Result<(), io::Error> {
    let mut error = String::new();

    if args.len() < ARGS_SIZE_LIMIT {
        error = String::from("not enough input argument");
    } else if args[1].is_empty() {
        error = String::from("filename is empty");
    }

    if error.is_empty() {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::NotFound, error))
    }
}

fn display_help(err: io::Error) {
    eprintln!("ERROR: {}\n", err);
    let help_message = String::from(
        "Usage:
Sample run using input data file",
    );
    println!("{}", help_message);
    std::process::exit(exitcode::USAGE);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // check the input arguments
    match argument_check(&args) {
        Ok(_) => print!("Reading data from path: "),
        Err(e) => display_help(e),
    };

    let input_file_path = Path::new(&args[1]);

    // Read data from file
    let (x_ptr, y_ptr) = match read_data::get_data(input_file_path) {
        Ok((x_ptr, y_ptr)) => (x_ptr, y_ptr),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    };

    // sample run of linear regression with data file
    linear_regression::sample_run(&x_ptr, &y_ptr);
}
