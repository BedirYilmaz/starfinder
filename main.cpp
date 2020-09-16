// main.cpp

#include <iostream>
#include "opencv2/core.hpp"
// #ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
        "{ help h |                  | Print help message. }"
        "{ input1 | box.png          | Path to input image 1. }"
        "{ input2 | box_in_scene.png | Path to input image 2. }";

const int MIN_MATCH_COUNT = 10;

int main( int argc, char* argv[] )
{
    // CommandLineParser parser( argc, argv, keys );

    char image_object_path[] = "/mnt/tera/code/cpp/projects/starfinder/images/Small_area.png";
    // char image_object_path[] = "/mnt/tera/code/cpp/projects/starfinder/images/Small_area_rotated.png";
    char image_scene_path[] = "/mnt/tera/code/cpp/projects/starfinder/images/StarMap.png";

    Mat img_object = imread( image_object_path, IMREAD_GRAYSCALE );
    Mat img_scene = imread( image_scene_path, IMREAD_GRAYSCALE );
    Mat img_matches;

    if ( img_object.empty() || img_scene.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

        
    // namedWindow("Simple Demo", WINDOW_AUTOSIZE);
    // imshow("Simple Demo", img_scene);
    // waitKey(0);
    // destroyAllWindows();

    for (int i =0; i<4; i++){
        
        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        int minHessian = 15;
        Ptr<SIFT> detector = SIFT::create();
        std::vector<KeyPoint> keypoints_object, keypoints_scene;
        Mat descriptors_object, descriptors_scene;
        detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
        detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.75f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        cout << good_matches.size() << endl;

        if (good_matches.size() < MIN_MATCH_COUNT){
            rotate(img_object, img_object, 90);
            continue;
        }    

        //-- Draw matches
        drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;

        for( size_t i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }

        if (obj.empty() || scene.empty()) {
            cout << "No matches found at all....." << endl;
            return 0;
        }


        

        Mat H = findHomography( obj, scene, RANSAC );

        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)img_object.cols, 0 );
        obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
        obj_corners[3] = Point2f( 0, (float)img_object.rows );
        std::vector<Point2f> scene_corners(4);

        
        

        perspectiveTransform( obj_corners, scene_corners, H);


        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
            scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
            scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
            scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
            scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );

        
        //-- Draw position labels to indicate the pixel coordinates of the bounding box

        putText(img_matches, "x: " + std::to_string((int)scene_corners[0].x) + " y: " + std::to_string((int)scene_corners[0].y), 
        scene_corners[0], FONT_HERSHEY_PLAIN, 2.0, CV_RGB(255,0,0), 2.0);

        //-- Show detected matches
        imshow("Good Matches & Object detection", img_matches );

        waitKey();
        return 0;
    }
}