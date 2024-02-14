use core::panic;
use std::env;
pub use std::path::Path;

use ml_rust::linear_regression;
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

    // sample run of linear regression with data file
    linear_regression::sample_run(&x_ptr, &y_ptr);

}
