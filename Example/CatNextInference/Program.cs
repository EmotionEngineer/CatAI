using System;
using System.Linq;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace CatNextInference
{
    public class Program
    {
        // Native method import from the catnext.dll library
        // This function performs the actual model inference
        [DllImport("catnext.dll", CallingConvention = CallingConvention.Cdecl)]
        public static unsafe extern void catnext_Inference(
            float* images,      // Pointer to 4D input tensor [batch, height, width, channels]
            float* info,        // Pointer to additional input data
            float* result       // Pointer to output prediction array
        );

        // Model configuration constants
        private const int Width = 384;          // Required frame width in pixels
        private const int Height = 384;         // Required frame height in pixels
        private const int FramesCount = 13;     // Number of frames needed for prediction
        private const float FrameInterval = 0.25f; // Time between frames in seconds (4 fps)
        private const int TotalChannels = FramesCount * 3; // Total color channels (13 frames × RGB)

        // Action label mapping: index-to-string for all 66 possible predictions
        private static readonly string[] ActionLabels = new string[66]
        {
            // Full list of cat actions
            "Poops", "Runs forward", "Interacts", "Moves forward and left at 45°", "Moves forward and left at 90°",
            "Moves forward and right at 45°", "Moves forward and right at 90°", "Moves forward and turns around", "Eats a mouse",
            "Climbs a tree", "Defends itself", "Plays with another cat", "Plays with a child",
            "Walks home", "Walks and meows", "Approaches a person", "Walks forward", "Digs a hole",
            "Eats", "Lies down and grooms itself", "Climbs", "Catches a mouse", "Jogs lightly", "Takes small steps",
            "Turns left 45°", "Turns left 45° and moves forward", "Turns left 90°", "Turns left 90° and moves forward", "Turns left and runs forward",
            "Attacks head-on", "Turns right 45°", "Turns right 45° and moves forward", "Turns right 90°", "Turns right 90° and moves forward",
            "Sniffs while walking", "Sniffs to the left", "Sniffs to the right", "Circles around obstacle to the left", "Widely circles around obstacle to the left",
            "Circles around obstacle to the right", "Widely circles around obstacle to the right", "Rests", "Jumps over", "Turns around",
            "Sneaks up", "Ascends", "Got through", "Got through and jumps", "Squeezes through", "Jumps down",
            "Jumps forward", "Jumps upward", "Drinks water", "Turns around and leaves",
            "Climbs down", "Stands and meows", "Stands and sniffs", "Stands and looks around",
            "Stands and glances up/down", "Stands and looks 180° left", "Stands and looks 90° left",
            "Stands and looks 180° right", "Stands and looks 90° right", "Stands and grooms itself",
            "Gets cuddled", "Walks over something"
        };

        public static void Main(string[] args)
        {
            try
            {
                // Validate input arguments
                if (args.Length == 0)
                    throw new ArgumentException("Usage: CatNextInference <video_path>");

                string videoPath = args[0];

                // Main processing pipeline:
                // 1. Process video file into input tensors
                // 2. Execute model inference
                // 3. Interpret and display results
                var (inputTensor, additionalInput) = ProcessVideoFile(videoPath);
                var prediction = RunModelInference(inputTensor, additionalInput);

                Console.WriteLine($"Predicted action: {GetActionLabel(prediction)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Environment.Exit(1);
            }
        }

        /// <summary>
        /// Main video processing pipeline
        /// </summary>
        private static (float[,,,], float[]) ProcessVideoFile(string videoPath)
        {
            // Open video file using OpenCV's VideoCapture
            using var videoCapture = new VideoCapture(videoPath);

            // Step 1: Validate video file properties
            ValidateVideoCapture(videoCapture);

            // Step 2: Extract relevant frames from video
            var frames = ExtractRelevantFrames(videoCapture);

            // Step 3: Convert frames to model input tensor format
            var tensor = PrepareInputTensor(frames);

            // Step 4: Create additional historical cat actions input
            // Even when no history exists, we must provide properly scaled values
            return (tensor, CreateAdditionalInput());
        }

        #region Video Processing

        /// <summary>
        /// Validates video file can be processed
        /// </summary>
        private static void ValidateVideoCapture(VideoCapture capture)
        {
            if (!capture.IsOpened())
                throw new Exception("Failed to open video file");

            if (capture.FrameCount < 1)
                throw new Exception("Video contains no frames");
        }

        /// <summary>
        /// Extracts key frames from video based on configured parameters
        /// </summary>
        private static Mat[] ExtractRelevantFrames(VideoCapture capture)
        {
            // Adjust FPS for invalid values
            double fps = GetAdjustedFps(capture);

            // Calculate frame sampling interval
            int frameStep = CalculateFrameStep(fps);

            // Determine starting frame position
            int startFrame = CalculateStartFrame(capture, frameStep);

            // Set video position to calculated start frame
            capture.Set(VideoCaptureProperties.PosFrames, startFrame);

            // Collect and process video frames
            return CollectFrames(capture, frameStep);
        }

        /// <summary>
        /// Ensures valid FPS value for calculations
        /// </summary>
        private static double GetAdjustedFps(VideoCapture capture)
        {
            double fps = capture.Fps;
            return fps < 1.0 ? 29.97 : fps; // Default to 29.97 for invalid FPS
        }

        /// <summary>
        /// Calculates frame sampling interval based on desired time between frames
        /// </summary>
        private static int CalculateFrameStep(double fps)
        {
            return (int)Math.Round(fps * FrameInterval);
        }

        /// <summary>
        /// Determines starting frame index to capture last 3.25 seconds
        /// </summary>
        private static int CalculateStartFrame(VideoCapture capture, int frameStep)
        {
            int totalFrames = (int)capture.FrameCount;
            return Math.Max(0, totalFrames - FramesCount * frameStep);
        }

        /// <summary>
        /// Collects video frames from specified capture with frame skipping
        /// </summary>
        /// <param name="capture">Initialized video capture object</param>
        /// <param name="frameStep">Number of frames to skip between captures</param>
        /// <returns>Array of processed video frames</returns>
        private static Mat[] CollectFrames(VideoCapture capture, int frameStep)
        {
            // Initialize frame buffer with target capacity
            Mat[] frames = new Mat[FramesCount];
            int collected = 0;

            // Default initialization to prevent null references
            Mat lastValidFrame = new Mat();  // Empty placeholder frame

            // Frame collection loop - runs until buffer full or video ends
            while (collected < FramesCount && capture.PosFrames < capture.FrameCount)
            {
                using var frame = new Mat();
                // Attempt to read and validate frame
                if (capture.Read(frame) && !frame.Empty())
                {
                    // Process frame and store in buffer
                    var processedFrame = ProcessFrame(frame);
                    frames[collected] = processedFrame;

                    // Clone frame to break dependency on video buffer
                    lastValidFrame = processedFrame.Clone();
                    collected++;
                }

                // Calculate and set next frame position
                capture.Set(VideoCaptureProperties.PosFrames, capture.PosFrames + frameStep);
            }

            // Fill missing frames using last valid frame (guaranteed non-null)
            HandleMissingFrames(frames, collected, lastValidFrame);

            // Clean up temporary frame storage
            lastValidFrame.Dispose();  // Release cloned frame resources

            return frames;
        }

        /// <summary>
        /// Processes individual frame: resize and color conversion
        /// </summary>
        private static Mat ProcessFrame(Mat frame)
        {
            var processed = new Mat();
            // Resize to model requirements
            Cv2.Resize(frame, processed, new Size(Width, Height));
            // Convert from BGR to RGB color space
            Cv2.CvtColor(processed, processed, ColorConversionCodes.BGR2RGB);
            return processed.Clone();
        }

        /// <summary>
        /// Populates empty frame slots with cloned copies of last valid frame
        /// </summary>
        private static void HandleMissingFrames(Mat[] frames, int collected, Mat lastFrame)
        {
            // Validate reference frame before cloning
            if (lastFrame == null || lastFrame.Empty())
            {
                throw new InvalidOperationException(
                    "Invalid reference frame for padding operation");
            }

            // Duplicate last valid frame for remaining slots
            for (int i = collected; i < FramesCount; i++)
            {
                frames[i] = lastFrame.Clone();  // Create independent copies
            }
        }

        #endregion

        #region Tensor Preparation

        /// <summary>
        /// Converts frame array to 4D input tensor [1, H, W, C]
        /// </summary>
        private static float[,,,] PrepareInputTensor(Mat[] frames)
        {
            var tensor = new float[1, Height, Width, TotalChannels];

            for (int frameIdx = 0; frameIdx < FramesCount; frameIdx++)
            {
                using var frame = frames[frameIdx];
                // Convert to 32-bit floating point format
                using var floatFrame = new Mat();
                frame.ConvertTo(floatFrame, MatType.CV_32FC3);

                // Add frame data to tensor
                AddFrameToTensor(floatFrame, tensor, frameIdx);
            }

            return tensor;
        }

        /// <summary>
        /// Copies frame data into tensor with channel-wise organization
        /// </summary>
        private static unsafe void AddFrameToTensor(Mat frame, float[,,,] tensor, int frameIndex)
        {
            int channelBase = frameIndex * 3; // Calculate channel offset
            var span = new Span<Vec3f>(frame.Data.ToPointer(), frame.Width * frame.Height);

            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    int index = y * Width + x;
                    var pixel = span[index];

                    // Normalize and store RGB values
                    tensor[0, y, x, channelBase] = pixel.Item0 / 255.0f;    // Red channel
                    tensor[0, y, x, channelBase + 1] = pixel.Item1 / 255.0f; // Green channel
                    tensor[0, y, x, channelBase + 2] = pixel.Item2 / 255.0f; // Blue channel
                }
            }
        }

        /// <summary>
        /// Creates additional input with proper normalization
        /// The additional input reflects the cat's 300 past actions
        /// Each action lasts 13 × 0.25 seconds ≈ 3.25 seconds
        /// 
        /// ModelAddInput[1][300] requirements:
        /// - Values represent historical actions in normalized form
        /// - Original range: -1 ("No info") to 65 (max action index)
        /// - Normalization formula: (value + 1) * scale
        ///   where scale = 1/66 ≈ 0.015151515151515152
        /// - This converts:
        ///   - "-1" (No info) → 0.0f
        ///   - Action indices 0-65 → [0.0151515, 1.0]
        /// </summary>
        private static float[] CreateAdditionalInput()
        {
            var input = new float[300];
            const int noInfoValue = -1;

            // Normalize "-1" (No info) to 0.0
            float normalizedValue = NormalizeAction(noInfoValue);

            Array.Fill(input, normalizedValue); // Fill for "no history"
            return input;
        }

        /// <summary>
        /// Normalizes action IDs to the model's expected input range [0.0, 1.0]
        /// </summary>
        static float NormalizeAction(int actionID)
            => actionID == -1 ? 0.0f : (actionID + 1) * (1.0f / 66.0f);

        #endregion

        #region Model Inference

        /// <summary>
        /// Executes model inference using native DLL
        /// </summary>
        private static unsafe float[] RunModelInference(float[,,,] input, float[] additionalInput)
        {
            float[] result = new float[66];

            // Pin memory for native interop
            fixed (float* inputPtr = input)
            fixed (float* addInputPtr = additionalInput)
            fixed (float* resultPtr = result)
            {
                catnext_Inference(inputPtr, addInputPtr, resultPtr);
            }

            return result;
        }

        /// <summary>
        /// Finds highest probability action from predictions
        /// </summary>
        private static string GetActionLabel(float[] predictions)
        {
            int maxIndex = 0;
            float maxValue = predictions[0];

            // Find index of maximum value
            for (int i = 1; i < predictions.Length; i++)
            {
                if (predictions[i] > maxValue)
                {
                    maxValue = predictions[i];
                    maxIndex = i;
                }
            }

            return ActionLabels[maxIndex];
        }

        #endregion
    }
}