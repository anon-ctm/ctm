// Get the specific video element by its ID
const videoElement = document.getElementById('scroll-play-video');
// Check if the video element exists
if (videoElement) {
    // Options for the Intersection Observer
    // threshold: 0.5 means the callback runs when 50% of the video is visible
    const options = {
        root: null, // Use the viewport as the root
        rootMargin: '0px',
        threshold: 0.5 // Trigger when 50% of the element is visible
    };
    // Callback function to execute when intersection changes
    const callback = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Video is visible - play it
                // Using a Promise to handle potential play() errors
                const playPromise = videoElement.play();
                if (playPromise !== undefined) {
                    playPromise.then(_ => {
                        // Autoplay started!
                        console.log("Video playback started on scroll.");
                    }).catch(error => {
                        // Autoplay was prevented.
                        console.error("Video playback failed:", error);
                        // Optionally show a play button or message to the user
                    });
                }
            } else {
                // Video is not visible - pause it
                videoElement.pause();
                console.log("Video playback paused.");
            }
        });
    };
    // Create the Intersection Observer
    const observer = new IntersectionObserver(callback, options);
    // Start observing the video element
    observer.observe(videoElement);
} else {
    console.error("Video element with ID 'scroll-play-video' not found.");
}
// Add error handling for video loading
document.querySelectorAll('video').forEach(video => {
    video.onerror = function() {
        console.error("Error loading video:", video.currentSrc);
        // Optionally display a message to the user
    };
     // Fallback for poster image if video fails to load
     if (video.poster) {
        video.onerror = () => {
            console.error(`Failed to load video: ${video.src}`);
            // Keep the poster visible or replace with an error message/image
            video.style.display = 'none'; // Hide broken video element
            const errorImg = document.createElement('img');
            errorImg.src = video.poster; // Show poster as fallback
            errorImg.alt = "Video failed to load";
            errorImg.style.maxWidth = '100%';
            errorImg.style.height = 'auto';
            errorImg.style.display = 'block';
            errorImg.style.margin = 'auto';
            video.parentNode.insertBefore(errorImg, video.nextSibling);
        };
    }
});