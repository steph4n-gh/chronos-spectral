# Sample video attribution

All real-camera calibration clips in `web/samples/` are transcoded to `640x360` H.264 `yuv420p`:

- `juggling_ballistic.mp4`: Cropped and scaled adaptation from `File:Zeitsymmetrie (jonglieren).webm` by Mathe-AC Lernwerkstatt Aachen on Wikimedia Commons (`https://commons.wikimedia.org/wiki/File:Zeitsymmetrie_(jonglieren).webm`), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Under ShareAlike terms, this cropped video adaptation remains licensed under CC BY-SA 4.0.
- `newton_cradle_impulses.mp4`: Time-normalized to real-time 60 fps from `File:Pendolo di Newton a 5 sferette rallentato.webm` on Wikimedia Commons (`https://commons.wikimedia.org/wiki/File:Pendolo_di_Newton_a_5_sferette_rallentato.webm`), dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
- `pets_pedestrian_occlusion.mp4`: Transcoded from `samples/data/vtest.avi` in the OpenCV repository (`https://github.com/opencv/opencv/blob/5.x/samples/data/vtest.avi`). Note: OpenCV distributes its repository under Apache-2.0, while the underlying footage originates from the academic PETS 2009 S2.L1 pedestrian tracking benchmark.
- `cup_hand_tremor.mp4`: Transcoded from `doc/js_tutorials/js_assets/cup.mp4` in the OpenCV repository (`https://github.com/opencv/opencv/blob/5.x/doc/js_tutorials/js_assets/cup.mp4`), originating from the BoBot (`Vid_B_cup`) / OpenCV tracking tutorial sequence.
- `webcam_book_occlusion.mp4`: Transcoded from `testdata/cv/tracking/faceocc2/data/faceocc2.webm` in `opencv_extra` (`https://github.com/opencv/opencv_extra/blob/4.x/testdata/cv/tracking/faceocc2/data/faceocc2.webm`), originating from the academic MILTrack / OTB `faceocc2` benchmark (`opencv_extra/testdata/cv/tracking/README.md`).

Before publishing a public open-source repository, verify whether you want to retain the CC BY-SA 4.0 clip and academic benchmark test clips in-tree or fetch them via script on demand.
