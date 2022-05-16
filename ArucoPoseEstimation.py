import math
import sys
import cv2
import numpy as np


def ips2mph(v_in_s):  # convert inches per second to miles per hour
    return v_in_s / 17.6


def velocity(framepersecond, tvec_prev, tvec_cur):  # calculate velocity
    time = 1.0 / framepersecond
    cur_speedx = (tvec_cur[0][0] - tvec_prev[0][0]) / time
    cur_speedy = (tvec_cur[1][0] - tvec_prev[1][0]) / time
    cur_speedz = (tvec_cur[2][0] - tvec_prev[2][0]) / time
    return np.array([cur_speedx, cur_speedy, cur_speedz])


def momentum(mass_kg, velocity_cm_s):  # calculate momentum
    return mass_kg * (velocity / 100)


def ave_velocity(start, end, total, speed, fps):  # average velocity
    assert end >= start
    assert end <= total
    assert start >= 1
    time = 1.0 / fps
    i = 0
    loc_start = np.zeros(3)
    loc_end = np.zeros(3)
    while i <= end - 1:
        if i <= start - 1:
            loc_start += speed[i] * time
        loc_end += speed[i] * time
        i += 1

    total_time = time * (end - start)
    ave_speed = (loc_end - loc_start) / total_time
    return ave_speed


class item:
    def __init__(self, name, aruco_id, shape, mass_kg):  # item class
        self.name = name
        self.aruco_id = aruco_id
        self.shape = shape
        self.mass_kg = mass_kg


def main():

    # Define an empty custom dictionary for markers of size 4X4
    aruco_dict = cv2.aruco.custom_dictionary(0, 4, 1)
    aruco_dict.bytesList = np.empty(shape=(3, 2, 4), dtype=np.uint8)

    # Add new marker(s)
    liquid_container_id = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]], dtype=np.uint8)
    fragile_sculpture_id = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
    empty_package_id = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)

    aruco_dict.bytesList[0] = cv2.aruco.Dictionary_getByteListFromBits(liquid_container_id)
    aruco_dict.bytesList[1] = cv2.aruco.Dictionary_getByteListFromBits(fragile_sculpture_id)
    aruco_dict.bytesList[2] = cv2.aruco.Dictionary_getByteListFromBits(empty_package_id)

    # Instatiate 3 physical objects
    liquid_container = item(name="liquid_container", aruco_id=0, shape="cube", mass_kg=1)
    fragile_sculpture = item(name="fragile_sculpture", aruco_id=1, shape="pyrimid", mass_kg=0.5)
    empty_package = item(name="empty_package", aruco_id=2, shape="rectangle", mass_kg=0.1)

    items = [liquid_container, fragile_sculpture, empty_package]

    # Read images from a video file in the current folder.
    video_capture = cv2.VideoCapture(0)  # Open video capture object
    got_image, bgr_image = video_capture.read()  # Make sure we can read video
    # video_capture = cv2.VideoCapture("1mph.mp4") # Open video capture object
    # got_image, bgr_image = video_capture.read() # Make sure we can read video

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

    if int(major_ver) < 3:
        fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Camera parameters
    fx = 497.55489548  # get these from calibration script
    fy = 496.59092265
    cx = 307.43577009
    cy = 243.13970998

    MARKER_LENGTH = 3.14961  # 8cm

    VEC_SCALE = 7

    # Camera intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(float)
    dist_coeff = np.array([[-0.01205136, 0.10580895, 0.0047374, -0.00269206, -0.19764752]])

    if not got_image:
        print("Cannot read video source")
        sys.exit()

    image_height, image_width = bgr_image.shape[:2]
    image_center = (image_width / 2, image_height / 2)
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    videoWriter = cv2.VideoWriter(
        "CSCI_507_Project.avi",
        fourcc=fourcc,
        fps=fps,
        frameSize=(image_width, image_height),
    )

    ll = MARKER_LENGTH / 2

    pyrimid_verticies = np.array([[-ll, -ll, 0], [ll, -ll, 0], [ll, ll, 0], [-ll, ll, 0], [0, 0, MARKER_LENGTH]])

    # append a one to end of each pyrimid verticie vector
    pyrimid_verticies = np.insert(pyrimid_verticies, 3, 1, axis=1)

    box_verticies = np.array(
        [
            [ll, ll, -ll, -ll, ll, ll, -ll, -ll],
            [-ll, ll, ll, -ll, -ll, ll, ll, -ll],
            [MARKER_LENGTH, MARKER_LENGTH, MARKER_LENGTH, MARKER_LENGTH, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    R_O_T = np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(math.radians(-90)), -np.sin(math.radians(-90))],
            [0, np.sin(math.radians(-90)), np.cos(math.radians(-90))],
        ]
    )
    T_O_T = np.array([0, -ll, -ll])

    n = 0
    nn = 0
    speed = []
    trajectory = []
    tvec_prev = None
    overlay_image = None
    object_color = (0, 255, 0)

    while True:
        overspeed = False
        vel = [0, 0, 0]
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop

        # Detect a marker.  Returns:
        # corners:   list of detected marker corners; for each marker, corners are clockwise)
        # ids:   vector of ids for the detected markers
        corners, ids, _ = cv2.aruco.detectMarkers(image=bgr_image, dictionary=aruco_dict)
        tag_center = None
        if corners:  # calculate center of tag
            x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
            y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]
            tag_center = (int(x_sum * 0.25), int(y_sum * 0.25))

        # pose of marker wrt camera
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners=corners,
            markerLength=MARKER_LENGTH,
            cameraMatrix=K,
            distCoeffs=dist_coeff,
        )

        if ids is not None:  # ensure we detect a tag
            rvec_m_c = rvecs[0]  # This is a 1x3 rotation vector
            tm_c = tvecs[0]  # This is a 1x3 translation vector

            # Homogeneous matrix of tag wrt camera
            R_T_C, _ = cv2.Rodrigues(rvec_m_c)
            R_T_C = R_T_C @ R_O_T
            T_T_C = (tm_c + T_O_T).T
            # T_T_C = tm_c.T

            if nn > 0:
                vel = velocity(fps, tvec_prev, T_T_C)
                vel_mag = np.linalg.norm(vel)
                color_scale = round(15 * vel_mag)
                if color_scale < 128:
                    object_color = (0, 255, 2 * color_scale)
                if 128 <= color_scale <= 254:
                    object_color = (0, 255 - color_scale, 255)
                if color_scale > 255:
                    overspeed = True
                    object_color = (0, 0, 255)

                name = None
                for obj in items:
                    if obj.aruco_id == ids[0][0]:
                        name = obj.name
                cv2.putText(
                    bgr_image,
                    "Object ID = " + name + " detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    color=object_color,
                    thickness=2,
                )
                cv2.putText(
                    bgr_image,
                    "Velocity_X = " + str(round(vel[0], 3)) + "in/s",
                    (50, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    color=object_color,
                    thickness=2,
                )
                cv2.putText(
                    bgr_image,
                    "Velocity_Y = " + str(round(vel[1], 3)) + "in/s",
                    (50, 90),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    color=object_color,
                    thickness=2,
                )
                cv2.putText(
                    bgr_image,
                    "Velocity_Z = " + str(round(vel[2], 3)) + "in/s",
                    (50, 110),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    color=object_color,
                    thickness=2,
                )
                # cv2.arrowedLine(bgr_image, tag_center, (tag_center[0] + int(vel[0]*VEC_SCALE), tag_center[1] + int(vel[1]*VEC_SCALE)), object_color,  2)
                speed.append(vel_mag)

            tvec_prev = T_T_C

            H_mat = np.block([[R_T_C, T_T_C], [0.0, 0.0, 0.0, 1.0]])

            M_ext = H_mat[0:3, :]

            if ids[0][0] == 0:  # This is a Box
                overlay_image = cv2.imread("liquid_container.png")  # Read in overlay image
                imcor = K @ M_ext @ box_verticies
                x_im = np.rint(np.divide(imcor[0, :], imcor[2, :])).astype(int)
                y_im = np.rint(np.divide(imcor[1, :], imcor[2, :])).astype(int)
                for i in range(3):
                    cv2.line(
                        bgr_image,
                        (x_im[i], y_im[i]),
                        (x_im[i + 1], y_im[i + 1]),
                        object_color,
                        2,
                    )

                cv2.line(bgr_image, (x_im[3], y_im[3]), (x_im[0], y_im[0]), object_color, 2)

                for i in range(4, 7):
                    cv2.line(
                        bgr_image,
                        (x_im[i], y_im[i]),
                        (x_im[i + 1], y_im[i + 1]),
                        object_color,
                        2,
                    )

                cv2.line(bgr_image, (x_im[7], y_im[7]), (x_im[4], y_im[4]), object_color, 2)

                for i in range(4):
                    cv2.line(
                        bgr_image,
                        (x_im[i], y_im[i]),
                        (x_im[i + 4], y_im[i + 4]),
                        object_color,
                        2,
                    )

                # Annotate tag pose
                cv2.aruco.drawAxis(
                    image=bgr_image,
                    cameraMatrix=K,
                    rvec=rvec_m_c,
                    tvec=tm_c,
                    length=0.7 * MARKER_LENGTH,
                    distCoeffs=dist_coeff,
                )

            elif ids[0][0] == 1:  # This is a pyrimid
                if overspeed:
                    overlay_image = cv2.imread("slow.jpg")  # Read in overlay image
                else:
                    overlay_image = cv2.imread("pyrimid.jpg")  # Read in overlay image
                points_image = []
                for point in pyrimid_verticies:
                    p = K @ M_ext @ point
                    p = np.squeeze(np.asarray(p))
                    p = np.divide(p, p[2]).round()
                    p = np.delete(p, 2, 0)
                    points_image.append(p)

                points_image = np.asarray(points_image)
                points_image = points_image.astype(int)
                point_count = 0

                for point in points_image:
                    bgr_image = cv2.circle(
                        bgr_image,
                        (point[0], point[1]),
                        radius=2,
                        color=object_color,
                        thickness=-1,
                    )
                    if point_count < 4:
                        bgr_image = cv2.line(
                            bgr_image,
                            tuple(points_image[point_count]),
                            tuple(points_image[(point_count + 1) % 4]),
                            color=object_color,
                            thickness=2,
                        )
                        bgr_image = cv2.line(
                            bgr_image,
                            tuple(points_image[point_count]),
                            tuple(points_image[4]),
                            color=object_color,
                            thickness=2,
                        )
                    point_count += 1

            # Draw border around marker
            # cv2.aruco.drawDetectedMarkers(image=bgr_image, corners=corners, ids=ids, borderColor=(255, 0, 255))

            trajectory.append(tag_center)
            for center in trajectory:
                cv2.circle(bgr_image, (int(center[0]), int(center[1])), 4, object_color, -1)
            nn += 1

        if corners:

            pts1 = np.array([corners[0][0][0], corners[0][0][1], corners[0][0][3], corners[0][0][2]]).astype(int)

            overlay_image_width, overlay_image_height = overlay_image.shape[:2]

            pts1_ortho = np.array(
                [
                    [0, 0],
                    [overlay_image_width, 0],
                    [0, overlay_image_height],
                    [overlay_image_width, overlay_image_height],
                ]
            )  # Scale overlay image

            # Find the homography to map the input image to the orthophoto image.
            H1, _ = cv2.findHomography(srcPoints=pts1_ortho, dstPoints=pts1)

            overlay_warped = cv2.warpPerspective(overlay_image, H1, (image_width, image_height))

            cv2.fillConvexPoly(
                bgr_image, np.array([[pts1[0], pts1[2], pts1[3], pts1[1]]]), np.zeros(3)
            )  # Remove original rgb, fill with zeros

            bgr_image = bgr_image + overlay_warped  # Fill in with warped overlay

        if all(v != 0 for v in vel):
            cv2.arrowedLine(
                bgr_image,
                tag_center,
                (
                    tag_center[0] + int(vel[0] * VEC_SCALE),
                    tag_center[1] + int(vel[1] * VEC_SCALE),
                ),
                object_color,
                2,
            )
        # show image
        cv2.imshow("my image", bgr_image)
        videoWriter.write(bgr_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # write to video
        # Wait for xx msec (0 means wait till a keypress).

    print("average calculated speed was " + str(ips2mph(sum(speed) / len(speed))) + " mph")
    video_capture.release()
    cv2.destroyAllWindows()

    # release
    videoWriter.release()


if __name__ == "__main__":
    main()
