// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <thread>



#define RED CV_RGB(255,0,0)
#define GREEN CV_RGB(0,255,0)
#define BLUE CV_RGB(0,0,255)
#define YELLOW CV_RGB(255,255,0)
#define CYAN CV_RGB(0,255,255)
#define MAGENTA CV_RGB(255,0,255)
#define ORANGE CV_RGB(255,165,0)


class Proiect {
private:
	// cascade clasifiers
	CascadeClassifier face_cascade; // cascade clasifier object for face
	CascadeClassifier eyes_cascade; // cascade clasifier object for eyes
	CascadeClassifier mouth_cascade; // cascade clasifier object for mouth
	CascadeClassifier nose_cascade; // cascade clasifier object for nose
	
	// object propertites
	static constexpr int minFaceSize = 30; // generic size for face
	struct ObjectProperties {
		const float minHeight;
		const float maxHeight;
		const int minObjectSize;

		ObjectProperties(const float minHeight, const float maxHeight, const int minObjectSize = minFaceSize / 5) :
			minHeight(minHeight),
			maxHeight(maxHeight),
			minObjectSize(minObjectSize)
		{
		}

	};

	const ObjectProperties eyeProperties = { 0.2f, 0.55f };
	const ObjectProperties noseProperties = { 0.4f, 0.75f };
	const ObjectProperties mouthProperties = { 0.7f, 0.99f };


	// show intermediate results 
	bool enableDebug;
	inline void showDebug(Mat& frame, std::vector<cv::Point2f>& corners, cv::Scalar& color) const {
		if (!enableDebug)
			return;

		for (auto p : corners)
			circle(frame, p, 1, color, 2);
	}
	inline void showDebugRect(Mat& frame, Rect& rect) const {
		if (!enableDebug)
			return;

		rectangle(frame, rect, RED, 1, 8, 0);
	}

	// load cascade clasifiers
	bool cascadeClasifiersLoadedSuccess;
	inline bool loadCascadeClassifiers()
	{
		static const String face_cascade_name = "./models/haarcascade_frontalface_alt.xml";
		static const String eyes_cascade_name = "./models/haarcascade_eye_tree_eyeglasses.xml";
		static const String mouth_cascade_name = "./models/haarcascade_mcs_mouth.xml";
		static const String nose_cascade_name = "./models/haarcascade_mcs_nose.xml";

		if (!face_cascade.load(face_cascade_name))
		{
			perror("Error loading face cascades: ");
			return false;
		}
		if (!eyes_cascade.load(eyes_cascade_name))
		{
			perror("Error loading eyes cascades: ");
			return false;
		}
		if (!mouth_cascade.load(mouth_cascade_name))
		{
			perror("Error loading mouth cascades: ");
			return false;
		}
		if (!nose_cascade.load(nose_cascade_name))
		{
			perror("Error loading nose cascades: ");
			return false;
		}

		return true;
	}
	

	// corners detection 
	std::vector<cv::Point2f> applyCornerDetection(Mat& frame)
	{
		Mat source;
		cvtColor(frame, source, CV_BGR2GRAY);

		vector<Point2f> corners;
		constexpr int maxCorners = 20;
		constexpr double qualityLevel = 0.01;
		constexpr double minDistance = 3;
		constexpr int blockSize = 3;
		constexpr bool useHarrisDetector = true;
		constexpr double k = 0.04;

		goodFeaturesToTrack(
			source,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(),
			blockSize,
			useHarrisDetector,
			k
		);

		showDebug(frame, corners, GREEN);

		return corners;
	}

	// interest eyebriws points detection
	void detectEyebrowsInterestPoints(Mat& frame, const Rect& face, const Rect& eyeRect)
	{
		constexpr float percentageIncrease = 0.15f;
		constexpr float percentageHeightDecrease = 0.6f;

		Rect eyebrowRect = eyeRect;
		eyebrowRect.y -= eyeRect.height * percentageIncrease;
		eyebrowRect.x -= eyeRect.width * percentageIncrease;
		eyebrowRect.width += 2 * eyeRect.width * percentageIncrease;
		eyebrowRect.height -= eyebrowRect.height * percentageHeightDecrease;
		
		const float eyeRectCenterX = eyeRect.x + eyeRect.width / static_cast<float>(2);

		Mat eyebrowRectFrame = frame(eyebrowRect);

		auto corners = applyCornerDetection(eyebrowRectFrame);

		auto partitionIt = std::partition(
			corners.begin(), corners.end(),
			[=](const cv::Point2f& p) {
				return p.x + eyebrowRect.x < eyeRectCenterX;
			}
		);

		std::vector<cv::Point2f> leftPoints(corners.begin(), partitionIt);
		std::vector<cv::Point2f> rightPoints(partitionIt, corners.end());
		
		if (!leftPoints.empty()) {
			int sum = 0;
			int minX = INT_MAX;
			for (auto& p : leftPoints) {
				sum += p.y;
				if (minX > p.x)
					minX = p.x;
			}
			sum /= leftPoints.size();

			circle(eyebrowRectFrame, cv::Point(minX, sum), 1, CYAN, 2);
		}

		if (!rightPoints.empty()) {
			int sum = 0;
			int maxX = INT_MIN;
			for (auto& p : rightPoints) {
				sum += p.y;
				if (maxX < p.x)
					maxX = p.x;
			}
			sum /= rightPoints.size();
			
			circle(eyebrowRectFrame, cv::Point(maxX, sum), 1, CYAN, 2);
		}

		showDebugRect(frame, eyebrowRect);
	}

	void detectEyesInterestPoints(Mat& frame, const Mat& frame_gray, const Rect& face)
	{
		const float minHeight = eyeProperties.minHeight;
		const float maxHeight = eyeProperties.maxHeight;
		const int minObjectSize = eyeProperties.minObjectSize;

		Rect eyesRect;
		eyesRect.x = face.x;
		eyesRect.y = face.y + minHeight * face.height;
		eyesRect.width = face.width;
		eyesRect.height = (maxHeight - minHeight) * face.height;		

		Mat eyesROI = frame_gray(eyesRect);
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(eyesROI, eyes, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minObjectSize, minObjectSize));

		constexpr auto nrOfEyes = 2; // if there are detected more than 2 eyes
		if (eyes.size() > nrOfEyes) {
			std::sort(eyes.begin(), eyes.end(), [](const Rect& a, const Rect& b) {
				return a.area() > b.area();
				}
			);
			eyes.resize(nrOfEyes);
		}
		
		constexpr int widthExtend = 3;
		for (auto& eyeRect : eyes) {
			// translate
			eyeRect.x += eyesRect.x - widthExtend;
			eyeRect.y += eyesRect.y;
			eyeRect.width += 2 * widthExtend;

			Point2f center(eyeRect.x + eyeRect.width / 2, eyeRect.y + eyeRect.height / 2);

			Mat eyeRectFrame = frame(eyeRect);
			auto corners = applyCornerDetection(eyeRectFrame);


			// delete points that are not in the eye area
			const auto centerY = center.y;
			const float maxDeviation = eyeRect.height * 0.12;
			corners.erase(std::remove_if(
				corners.begin(), corners.end(),
				[=](const cv::Point2f& p) {
					return
						(p.y + eyeRect.y < centerY - maxDeviation) ||
						(centerY + maxDeviation < p.y + eyeRect.y);
				}
			), corners.end());

			// show valid points
			showDebug(eyeRectFrame, corners, YELLOW);

			std::sort(corners.begin(), corners.end(), [](const Point2f& a, const Point2f& b) {
				return a.x < b.x;
				}
			);

			circle(eyeRectFrame, corners.front(), 1, ORANGE, 2);
			circle(eyeRectFrame, corners.back(), 1, ORANGE, 2);

			circle(frame, center, 1, RED, 2);

			detectEyebrowsInterestPoints(frame, face, eyeRect);

			showDebugRect(frame, eyeRect);
		}
	}

	void detectNoseInterestPoints(Mat& frame,const Mat& frame_gray, const Rect& face)
	{
		const float minHeight = noseProperties.minHeight;
		const float maxHeight = noseProperties.maxHeight;
		const int minObjectSize = noseProperties.minObjectSize;

		Rect noseArea;
		noseArea.x = face.x;
		noseArea.y = face.y + minHeight * face.height;
		noseArea.width = face.width;
		noseArea.height = (maxHeight - minHeight) * face.height;

		Mat noseROI = frame_gray(noseArea);
		std::vector<Rect> nose;
		nose_cascade.detectMultiScale(noseROI, nose, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minObjectSize, minObjectSize));

		constexpr auto nrOfNoses = 1; // if there are detected more than 1 noses
		if (nose.size() > nrOfNoses) {
			std::sort(
				nose.begin(), nose.end(),
				[](const Rect& a, const Rect& b) {
					return a.area() > b.area();
				}
			);
			nose.resize(nrOfNoses);
		}

		for (auto& noseRect : nose) {
			// translate
			noseRect.x += noseArea.x;
			noseRect.y += noseArea.y;

			Mat noseRectFrame = frame(noseRect);

			Point2f center(noseRect.x + noseRect.width / 2, noseRect.y + noseRect.height / 2);

			auto corners = applyCornerDetection(frame(noseRect));

			// delete points that are not in the nose interrest area
			const auto centerX = center.x;
			const auto centerY = center.y;
			const float xDeviation = noseRect.width * 0.09;
			const float yDeviation = noseRect.height * 0.12;
			corners.erase(std::remove_if(
				corners.begin(), corners.end(),
				[=](const cv::Point2f& p) {
					const auto px = p.x;
					const auto py = p.y;
					return
						(py + noseRect.y < centerY + 2) ||
						(py > noseRect.height - yDeviation) ||
						(px < xDeviation) ||
						(px > noseRect.width - xDeviation) ||
						(abs(px + noseRect.x - centerX) < xDeviation)
						;
				}
			), corners.end());
			showDebug(noseRectFrame, corners, YELLOW);

			auto partitionIt = std::partition(
				corners.begin(), corners.end(),
				[=](const cv::Point2f& p) {
					return p.x + noseRect.x < centerX;
				}
			);

			std::vector<cv::Point2f> leftPoints(corners.begin(), partitionIt);
			std::vector<cv::Point2f> rightPoints(partitionIt, corners.end());
			
			auto computeAverage = [](std::vector<cv::Point2f>& points){
				cv::Point2f sumAvg(0, 0);
				for (const auto& p : points)
					sumAvg += p;
				sumAvg /= static_cast<float>(points.size());
				return sumAvg;
			};
			
			if (!leftPoints.empty()) {
				auto leftAverage = computeAverage(leftPoints);
				circle(noseRectFrame, leftAverage, 1, MAGENTA, 2);
			}

			if (!rightPoints.empty()) {
				auto rightAverage = computeAverage(rightPoints);
				circle(noseRectFrame, rightAverage, 1, MAGENTA, 2);
			}

			circle(frame, center, 1, MAGENTA, 2);
			
			showDebugRect(frame, noseRect);
		}
		
	}

	void detectMouthInterestPoints(Mat& frame, const Mat& frame_gray, const Rect& face)
	{
		const float minHeight = mouthProperties.minHeight;
		const float maxHeight = mouthProperties.maxHeight;
		const int minObjectSize = mouthProperties.minObjectSize;

		Rect mouthArea;
		mouthArea.x = face.x;
		mouthArea.y = face.y + minHeight * face.height;
		mouthArea.width = face.width;
		mouthArea.height = (maxHeight - minHeight) * face.height;

		Mat mouthROI = frame_gray(mouthArea);
		std::vector<Rect> mouth;
		mouth_cascade.detectMultiScale(mouthROI, mouth, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minObjectSize, minObjectSize));

		constexpr auto nrOfMouths = 1; // if there are detected more than 1 mouths
		if (mouth.size() > nrOfMouths) {
			std::sort(mouth.begin(), mouth.end(), [](const Rect& a, const Rect& b) {
				return a.area() > b.area();
			});
			mouth.resize(nrOfMouths);
		}

		constexpr int heightExtend = 5;
		constexpr int widthExtend = 5;
		for (auto& mouthRect : mouth) {
			// translate
			mouthRect.x += mouthArea.x - widthExtend;
			mouthRect.y += mouthArea.y - heightExtend;
			mouthRect.width += 2 * widthExtend;
			mouthRect.height += heightExtend;

			Mat mouthRectFrame = frame(mouthRect);

			Point2f center(mouthRect.x + mouthRect.width / 2, mouthRect.y + mouthRect.height / 2);

			auto corners = applyCornerDetection(frame(mouthRect));

			// delete points that are not in the mouth area
			const auto centerX = center.x;
			const auto centerY = center.y;
			corners.erase(std::remove_if(
				corners.begin(), corners.end(),
				[=](const cv::Point2f& p) {
					return (p.y + mouthRect.y > centerY); // remove above mouth area center
				}
			), corners.end());
			showDebug(mouthRectFrame, corners, YELLOW);

			std::sort(corners.begin(), corners.end(), [](const Point2f& a, const Point2f& b) {
				return a.x < b.x;
				}
			);

			circle(mouthRectFrame, corners.front(), 1, BLUE, 2);
			circle(mouthRectFrame, corners.back(), 1, BLUE, 2);

			const float xDeviation = mouthRect.width * 0.1;
			std::vector<cv::Point2f> midPoints;
			std::copy_if(
				corners.begin(), corners.end(),
				std::back_inserter(midPoints),
				[=](const cv::Point2f& p) {
					return abs(p.x + mouthRect.x - centerX) <= xDeviation;
				}
			);

			if (midPoints.size() > 0) {
				cv::Point2f sumAvg(0, 0);
				for (const auto& p : midPoints)
					sumAvg += p;
				sumAvg /= static_cast<float>(midPoints.size());
				circle(mouthRectFrame, sumAvg, 1, BLUE, 2);
			}

			circle(frame, center, 1, BLUE, 2);

			showDebugRect(frame, mouthRect);
		}

	}

	void detectInterestFacePoints(Mat& frame)
	{
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		// Detect faces
		std::vector<Rect> faces;
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(minFaceSize, minFaceSize));

		for (auto& face : faces)
		{
			showDebugRect(frame, face);

			std::thread eyesThread(
				[&]() {
					detectEyesInterestPoints(std::ref(frame), std::ref(frame_gray), face);
				}
			);

			std::thread noseThread(
				[&]() {
					detectNoseInterestPoints(std::ref(frame), std::ref(frame_gray), face);
				}
			);

			std::thread mouthThread(
				[&]() {
					detectMouthInterestPoints(std::ref(frame), std::ref(frame_gray), face);
				}
			);


			eyesThread.join();
			noseThread.join();
			mouthThread.join();

		}

	}

public:

	Proiect() :
		enableDebug(false)
	{
		cascadeClasifiersLoadedSuccess = loadCascadeClassifiers();
		if (!cascadeClasifiersLoadedSuccess)
			throw std::exception("Cascade Classifiers failed to load, check the console");
	}


	~Proiect() {
	}

	void beginProcess()
	{
		if (!cascadeClasifiersLoadedSuccess) {
			printf("ERROR: Cascade Classifiers failed to load");
			return;
		}

		char fileName[MAX_PATH];
		while (openFileDlg(fileName))
		{
			double t = (double)getTickCount();

			Mat source = imread(fileName, CV_LOAD_IMAGE_COLOR);
			Mat dst = source.clone();

			detectInterestFacePoints(dst);

			imshow("Punctele de interes", dst);

			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%.3f [ms]\n", t * 1000);

			waitKey();
		}
	}
	
	inline void showIntermediateResults(bool flag) {
		enableDebug = flag;
	}
	inline void enableIntermediateResults() {
		enableDebug = true;
	}
	inline void disableIntermediateResults() {
		enableDebug = false;
	}
};


int main()
{
	Proiect p;
	p.showIntermediateResults(false);
	p.beginProcess();

	return 0;
}