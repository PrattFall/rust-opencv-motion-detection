extern crate clap;

use chrono::offset::Utc;
use chrono::DateTime;
use clap::{App, Arg, SubCommand};
use opencv::{core, highgui, imgproc, prelude::*, types::VectorOfVectorOfPoint, videoio, Result};
use std::time::SystemTime;

fn draw_timestamp(src: &Mat, mut dst: &mut Mat, frame_diff: i32) -> Result<()> {
    src.copy_to(dst)?;

    let now = SystemTime::now();
    let nowdate: DateTime<Utc> = now.into();

    imgproc::put_text(
        &mut dst,
        &*format!(
            "{} | Frame Diff: {}",
            nowdate.format("%d/%m/%Y %T"),
            frame_diff
        ),
        core::Point::new(5, 30),
        imgproc::FONT_HERSHEY_DUPLEX,
        0.75,
        core::Scalar::new(255., 255., 255., 255.),
        1,
        1,
        false,
    )?;

    Ok(())
}

fn main() -> Result<()> {
    let matches = App::new("Video Capture OpenCV")
        .version("0.1")
        .author("Brian P. <bjp2693@gmail.com>")
        .about("Records video whenever motion is detected")
        .arg(Arg::with_name("filename")
             .short("f")
             .long("file")
             .value_name("FILE_LOCATION")
             .help("Sets the location and filename prefix of the output video")
             .takes_value(true))
        .arg(Arg::with_name("webcam")
             .short("w")
             .long("webcam")
             .value_name("WEBCAM_ID")
             .help("The id of the webcam to use (defaults to 0)")
             .takes_value(true))
        .arg(Arg::with_name("threshold")
             .short("t")
             .long("threshold")
             .value_name("THRESHOLD")
             .help("The motion detection threshold to use. Higher numbers detect motion more easily, but cause more false positives. (defaults to 10)")
             .takes_value(true))
        .get_matches();

    let window = "video capture";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let webcam_id = matches
        .value_of("webcam")
        .unwrap_or("0")
        .parse::<i32>()
        .expect(
        "The given webcam id must be an integer less than the number of webcams you have connected",
    );

    let mut cam = videoio::VideoCapture::new(webcam_id, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;

    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut first_frame = true;
    let mut previous_frame = Mat::default();
    let mut frame = Mat::default();
    let mut frame_diff = 0;
    cam.read(&mut frame)?;

    let file_location = matches.value_of("filename").unwrap_or("./test.avi");

    let mut writer = videoio::VideoWriter::new(
        file_location,
        videoio::VideoWriter::fourcc(77, 74, 80, 71)?, // 'MJPG'
        12.,
        frame.size()?,
        true,
    )?;

    loop {
        cam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            // Convert to Grayscale
            let mut gray = Mat::default();
            imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

            // Blur
            let mut blurred = Mat::default();
            imgproc::gaussian_blur(
                &gray,
                &mut blurred,
                core::Size::new(25, 25),
                0_f64,
                0_f64,
                0,
            )?;

            // Diff only if both frame and previous_frame actually exist
            if previous_frame.size()?.width > 0 {
                let mut diff = Mat::default();
                core::absdiff(&blurred, &previous_frame, &mut diff)?;

                let threshold = matches
                    .value_of("threshold")
                    .unwrap_or("10")
                    .parse::<f64>()
                    .expect("The threshold given must be a floating point number");

                let mut thresh = Mat::default();
                imgproc::threshold(
                    &diff,
                    &mut thresh,
                    threshold,
                    255_f64,
                    imgproc::THRESH_BINARY,
                )?;

                let mut contours = VectorOfVectorOfPoint::new();
                imgproc::find_contours(
                    &thresh,
                    &mut contours,
                    imgproc::RETR_EXTERNAL,
                    imgproc::CHAIN_APPROX_SIMPLE,
                    core::Point::new(0, 0),
                )?;

                if !(contours as VectorOfVectorOfPoint).is_empty() {
                    let mut timestamped = Mat::default();
                    draw_timestamp(&frame, &mut timestamped, frame_diff)?;

                    // Keep track of changes
                    previous_frame = blurred.clone();
                    writer.write(&timestamped)?;
                    highgui::imshow(window, &timestamped)?;
                    frame_diff = 0;
                }
            }

            // We need a previous frame to compare to in future iterations
            if first_frame {
                previous_frame = blurred.clone();
                first_frame = false;
            }

            frame_diff = frame_diff + 1;
        }

        // Stop on keypress
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    Ok(())
}
