
// Wrap everything in a function that can be called from HTML
function initializeImageNetPlayer(videoFiles, config = {}) {

    // --- Configuration (use provided config or defaults) ---
    const videoBasePath = config.videoBasePath || 'assets/imagenet_mp4s/';
    const thumbnailBasePath = config.thumbnailBasePath || 'assets/imagenet_mp4s/thumbnails/';
    const playerElementId = config.playerElementId || 'main-video';
    const thumbnailGridId = config.thumbnailGridId || 'thumbnail-grid';
    // --- End Configuration ---

    // Wait for the DOM to be ready before manipulating it
    document.addEventListener('DOMContentLoaded', () => {

        // Check if the required videoFiles list was provided
        if (!videoFiles || videoFiles.length === 0) {
            console.error("Initialization Error: No video files provided to initializeImageNetPlayer.");
            // Display an error message to the user in the thumbnail area
            const grid = document.getElementById(thumbnailGridId);
            if(grid) {
                grid.innerHTML = '<p style="color: red;">Error: Video list not provided.</p>';
            }
            return; // Stop execution if no videos
        }

        const thumbnailGrid = document.getElementById(thumbnailGridId);
        if (!thumbnailGrid) {
             console.error(`Initialization Error: Thumbnail grid element with ID '${thumbnailGridId}' not found.`);
             return;
        }

        let player; // Variable to hold the Video.js player instance

        // Video.js Options (merge defaults with any provided in config)
        const defaultVideoOptions = {
            controls: true,
            autoplay: 'muted',
            loop: true,
            fluid: false,
            playbackRates: [0.1, 0.5, 1, 1.5, 2, 3],
            // Set background color directly in options if desired
            // backgroundColor: '#ffffff' // Alternative to CSS method
        };
        const videoOptions = { ...defaultVideoOptions, ...(config.videojsOptions || {}) };


        // Check if the player element exists before initializing
        const playerElement = document.getElementById(playerElementId);
        if (!playerElement) {
            console.error(`Initialization Error: Video player element with ID '${playerElementId}' not found.`);
             return;
        }

        // Initialize Video.js Player
        player = videojs(playerElementId, videoOptions, function onPlayerReady() {
            videojs.log('Player is ready');
             this.options(videoOptions); // Ensure options are applied

            // --- Player Event Listeners ---
            this.on('ended', function() {
                videojs.log('Video ended (will loop if loop:true)');
            });
            this.on('error', function() {
                const error = this.error();
                videojs.log(`ERROR: Player error code ${error?.code}, ${error?.message}`);
            });
            // --- End Player Event Listeners ---

            loadInitialRandomVideo(); // Load initial video when ready
        });

        // --- Core Functions ---

        function loadVideo(baseName) {
            const videoSrc = `${videoBasePath}${baseName}.mp4`;
            console.log(`Loading video via Video.js: ${videoSrc}`);
            if (player && typeof player.src === 'function') {
                 const currentSrc = player.currentSrc();
                 if (currentSrc && currentSrc.endsWith(videoSrc)) {
                     console.log("Video.js: Source already loaded.");
                     if (player.paused()) { player.play().catch(handlePlayError); }
                     return;
                 }
                player.src({ type: 'video/mp4', src: videoSrc });
                player.play().catch(handlePlayError);
            } else {
                console.error("Video.js player not available in loadVideo.");
            }
        }

        function handlePlayError(error) {
            if (error.name === 'NotAllowedError') {
                 console.warn("Video.js: Playback prevented (ensure user interaction or muted autoplay).");
            } else {
                 console.error("Video.js: Error attempting to play video:", error);
            }
        }

        function createThumbnails() {
            thumbnailGrid.innerHTML = ''; // Clear any existing thumbnails
            videoFiles.forEach(baseName => {
                const img = document.createElement('img');
                img.src = `${thumbnailBasePath}${baseName}.jpg`;
                img.alt = `Thumbnail for video ${baseName}`;
                img.classList.add('thumbnail-button');
                img.dataset.videoBaseName = baseName;
                img.tabIndex = 0;

                // Fallback for missing thumbnails
                img.onerror = function() {
                    img.alt = `Thumbnail missing for ${baseName}`;
                    // Optional: Display placeholder text or style differently
                    img.style.border = '1px dashed red';
                    console.warn(`Thumbnail not found: ${img.src}`);
                };

                img.addEventListener('click', () => { loadVideo(baseName); });
                img.addEventListener('keypress', (event) => {
                     if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        loadVideo(baseName);
                     }
                });
                thumbnailGrid.appendChild(img);
            });
        }

        function loadInitialRandomVideo() {
            if (videoFiles.length > 0) {
                const randomIndex = Math.floor(Math.random() * videoFiles.length);
                const initialVideoBaseName = videoFiles[randomIndex];
                loadVideo(initialVideoBaseName);
            } else {
                console.warn("Cannot load initial video: videoFiles array is empty.");
            }
        }

        // --- End Core Functions ---

        // --- Initial Setup Calls ---
        createThumbnails();
        // NOTE: Initial video loading is triggered inside the player.ready() callback
        // --- End Initial Setup Calls ---

    }); // End DOMContentLoaded listener

} // End initializeImageNetPlayer function

// Optional: Make the function globally available if not using modules
// window.initializeImageNetPlayer = initializeImageNetPlayer;