use chrono::offset::Utc;
use chrono::DateTime;
use opencv::{core, highgui, imgproc, prelude::*, types::VectorOfVectorOfPoint, videoio, Result};
use std::time::SystemTime;

fn main() -> Result<()> {
    let window = "video capture";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;

    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut first_frame = true;
    let mut previous_frame = Mat::default();
    let mut frame = Mat::default();
    let mut frame_diff = 0;
    cam.read(&mut frame)?;

    let mut writer = videoio::VideoWriter::new(
        "test.avi",
        videoio::VideoWriter::fourcc(77, 74, 80, 71)?, // 'MJPG'
        24.,
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

                let mut thresh = Mat::default();
                imgproc::threshold(&diff, &mut thresh, 10_f64, 255_f64, imgproc::THRESH_BINARY)?;

                let mut contours = VectorOfVectorOfPoint::new();

                imgproc::find_contours(
                    &thresh,
                    &mut contours,
                    imgproc::RETR_EXTERNAL,
                    imgproc::CHAIN_APPROX_SIMPLE,
                    core::Point::new(0, 0),
                )?;

                if !(contours as VectorOfVectorOfPoint).is_empty() {
                    let mut timestamped = frame.clone();
                    let now = SystemTime::now();
                    let nowdate: DateTime<Utc> = now.into();

                    imgproc::put_text(
                        &mut timestamped,
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
