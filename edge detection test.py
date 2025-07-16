import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os


class EdgeDetectionComparison:
    def __init__(self, image_path):
        """
        Initialize the edge detection comparison tool

        Args:
            image_path (str): Path to the input image
        """
        self.image_path = image_path
        self.original_image = None
        self.gray_image = None
        self.load_image()

    def load_image(self):
        """Load and preprocess the image"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # Read the image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {self.image_path}")

        # Convert BGR to RGB for matplotlib display
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for edge detection
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

    def apply_canny(self, low_threshold=50, high_threshold=150):
        """
        Apply Canny edge detection

        Args:
            low_threshold (int): Lower threshold for edge detection
            high_threshold (int): Upper threshold for edge detection

        Returns:
            numpy.ndarray: Canny edge detection result
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 1.4)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    def apply_sobel(self, kernel_size=3):
        """
        Apply Sobel edge detection

        Args:
            kernel_size (int): Size of the Sobel kernel (must be odd)

        Returns:
            numpy.ndarray: Sobel edge detection result
        """
        # Apply Sobel operator in X and Y directions
        sobel_x = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # Combine X and Y gradients
        sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Normalize to 0-255 range
        sobel_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))

        return sobel_normalized

    def compare_algorithms(self, canny_low=50, canny_high=150, sobel_kernel=3):
        """
        Compare Canny and Sobel edge detection side by side

        Args:
            canny_low (int): Canny lower threshold
            canny_high (int): Canny upper threshold
            sobel_kernel (int): Sobel kernel size
        """
        # Apply both algorithms
        canny_edges = self.apply_canny(canny_low, canny_high)
        sobel_edges = self.apply_sobel(sobel_kernel)

        # Create subplot figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(self.original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Canny edges
        axes[1].imshow(canny_edges, cmap='gray')
        axes[1].set_title(f'Canny Edge Detection\n(Low: {canny_low}, High: {canny_high})')
        axes[1].axis('off')

        # Sobel edges
        axes[2].imshow(sobel_edges, cmap='gray')
        axes[2].set_title(f'Sobel Edge Detection\n(Kernel: {sobel_kernel}x{sobel_kernel})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    def interactive_comparison(self):
        """
        Create an interactive comparison with sliders
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)

        # Initial parameters
        canny_low = 50
        canny_high = 150
        sobel_kernel = 3

        # Apply initial edge detection
        canny_edges = self.apply_canny(canny_low, canny_high)
        sobel_edges = self.apply_sobel(sobel_kernel)

        # Display images
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.gray_image, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')

        im_canny = axes[1, 0].imshow(canny_edges, cmap='gray')
        axes[1, 0].set_title('Canny Edge Detection')
        axes[1, 0].axis('off')

        im_sobel = axes[1, 1].imshow(sobel_edges, cmap='gray')
        axes[1, 1].set_title('Sobel Edge Detection')
        axes[1, 1].axis('off')

        # Create sliders
        ax_canny_low = plt.axes([0.2, 0.15, 0.5, 0.03])
        ax_canny_high = plt.axes([0.2, 0.1, 0.5, 0.03])
        ax_sobel_kernel = plt.axes([0.2, 0.05, 0.5, 0.03])

        slider_canny_low = Slider(ax_canny_low, 'Canny Low', 0, 200, valinit=canny_low)
        slider_canny_high = Slider(ax_canny_high, 'Canny High', 0, 300, valinit=canny_high)
        slider_sobel_kernel = Slider(ax_sobel_kernel, 'Sobel Kernel', 1, 7, valinit=sobel_kernel, valstep=2)

        def update(val):
            new_canny_low = int(slider_canny_low.val)
            new_canny_high = int(slider_canny_high.val)
            new_sobel_kernel = int(slider_sobel_kernel.val)

            # Ensure kernel size is odd
            if new_sobel_kernel % 2 == 0:
                new_sobel_kernel += 1

            # Update edge detection
            new_canny_edges = self.apply_canny(new_canny_low, new_canny_high)
            new_sobel_edges = self.apply_sobel(new_sobel_kernel)

            # Update images
            im_canny.set_array(new_canny_edges)
            im_sobel.set_array(new_sobel_edges)

            # Update titles
            axes[1, 0].set_title(f'Canny (Low: {new_canny_low}, High: {new_canny_high})')
            axes[1, 1].set_title(f'Sobel (Kernel: {new_sobel_kernel}x{new_sobel_kernel})')

            fig.canvas.draw()

        # Connect sliders to update function
        slider_canny_low.on_changed(update)
        slider_canny_high.on_changed(update)
        slider_sobel_kernel.on_changed(update)

        plt.show()


def main():
    """
    Main function to run the edge detection comparison
    """
    # Example usage
    print("Edge Detection Comparison Tool")
    print("=" * 40)

    # Get image path from user
    image_path = input("Enter the path to your image: ").strip()

    try:
        # Create comparison object
        detector = EdgeDetectionComparison(image_path)

        # Show basic comparison
        print("\nShowing basic comparison...")
        detector.compare_algorithms()

        # Ask if user wants interactive comparison
        choice = input("\nWould you like to try interactive comparison with sliders? (y/n): ").strip().lower()

        if choice == 'y':
            print("Starting interactive comparison...")
            detector.interactive_comparison()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required libraries installed:")
        print("pip install opencv-python matplotlib numpy")


# Example usage for testing
if __name__ == "__main__":
    # You can also use it directly like this:
    # detector = EdgeDetectionComparison("path/to/your/image.jpg")
    # detector.compare_algorithms(canny_low=50, canny_high=150, sobel_kernel=3)
    # detector.interactive_comparison()

    main()