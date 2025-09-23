// AI Co-Creator - Enhanced UI with Full Functionality
class AICoCreatorManualApp {
    constructor() {
        this.apiUrl = 'http://localhost:8000/api/v1';
        this.videos = [];
        this.selectedVideo = null;
        this.processingJobs = new Map();
        this.renderJobs = new Map();
        this.contentData = null;
        this.timeline = null;
        this.platformPresets = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkSystemStatus();
        this.loadVideos();
        this.startStatusPolling();
    }

    setupEventListeners() {
        this.setupNavigation();
        this.setupVideoLibrary();
        this.setupPipeline();
        this.setupEditor();
        this.setupRenderer();
        this.setupUpload();
        this.setupRefresh();
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const tabContents = document.querySelectorAll('.tab-content');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();

                // Update active states
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');

                // Show correct tab
                tabContents.forEach(content => content.classList.add('hidden'));
                const tabId = link.dataset.tab + '-tab';
                const targetTab = document.getElementById(tabId);
                if (targetTab) {
                    targetTab.classList.remove('hidden');

                    // Tab-specific initialization
                    this.onTabChange(link.dataset.tab);
                }
            });
        });
    }

    onTabChange(tab) {
        switch(tab) {
            case 'library':
                this.loadVideos();
                break;
            case 'pipeline':
                this.updatePipelineVideoSelector();
                break;
            case 'editor':
                this.updateEditorVideoSelector();
                break;
            case 'renderer':
                this.updateRendererVideoSelector();
                this.loadPlatformPresets();
                this.loadRendererData();
                // Start periodic refresh for renderer data
                this.startRendererDataRefresh();
                break;
        }
    }

    setupVideoLibrary() {
        const loadVideosBtn = document.getElementById('load-videos-btn');
        if (loadVideosBtn) {
            loadVideosBtn.addEventListener('click', () => this.loadVideos());
        }
    }

    setupPipeline() {
        // Shot Detection
        const startShotDetection = document.getElementById('start-shot-detection');
        if (startShotDetection) {
            startShotDetection.addEventListener('click', () => this.startShotDetection());
        }

        const shotMethod = document.getElementById('shot-method');
        if (shotMethod) {
            shotMethod.addEventListener('change', () => this.updateShotThreshold());
        }

        const clearShots = document.getElementById('clear-shots');
        if (clearShots) {
            clearShots.addEventListener('click', () => this.clearData('shots'));
        }

        // Transcription
        const startTranscription = document.getElementById('start-transcription');
        if (startTranscription) {
            startTranscription.addEventListener('click', () => this.startTranscription());
        }

        const clearTranscripts = document.getElementById('clear-transcripts');
        if (clearTranscripts) {
            clearTranscripts.addEventListener('click', () => this.clearData('transcripts'));
        }

        // Analysis
        const startAnalysis = document.getElementById('start-analysis');
        if (startAnalysis) {
            startAnalysis.addEventListener('click', () => this.startAnalysis());
        }

        const clearAnalysis = document.getElementById('clear-analysis');
        if (clearAnalysis) {
            clearAnalysis.addEventListener('click', () => this.clearData('analysis'));
        }

        // Pipeline actions
        const runFullPipeline = document.getElementById('run-full-pipeline');
        if (runFullPipeline) {
            runFullPipeline.addEventListener('click', () => this.runFullPipeline());
        }

        const getVideoStatus = document.getElementById('get-video-status');
        if (getVideoStatus) {
            getVideoStatus.addEventListener('click', () => this.getVideoStatus());
        }

        const clearAllData = document.getElementById('clear-all-data');
        if (clearAllData) {
            clearAllData.addEventListener('click', () => this.clearAllData());
        }

        // Video selector
        const pipelineVideoSelect = document.getElementById('pipeline-video-select');
        if (pipelineVideoSelect) {
            pipelineVideoSelect.addEventListener('change', (e) => {
                const videoId = parseInt(e.target.value);
                if (videoId) {
                    this.selectVideoForPipeline(videoId);
                }
            });
        }
    }

    setupEditor() {
        const generateContent = document.getElementById('generate-content');
        if (generateContent) {
            generateContent.addEventListener('click', () => this.generateContent());
        }

        const regenerateContent = document.getElementById('regenerate-content');
        if (regenerateContent) {
            regenerateContent.addEventListener('click', () => this.regenerateContent());
        }

        const editorVideoSelect = document.getElementById('editor-video-select');
        if (editorVideoSelect) {
            editorVideoSelect.addEventListener('change', (e) => {
                const videoId = parseInt(e.target.value);
                if (videoId) {
                    this.selectVideoForEditor(videoId);
                }
            });
        }
    }

    setupRenderer() {
        const startRender = document.getElementById('start-render');
        if (startRender) {
            startRender.addEventListener('click', () => this.startRender());
        }

        const addToQueue = document.getElementById('add-to-queue');
        if (addToQueue) {
            addToQueue.addEventListener('click', () => this.addToRenderQueue());
        }

        const rendererVideoSelect = document.getElementById('renderer-video-select');
        if (rendererVideoSelect) {
            rendererVideoSelect.addEventListener('change', (e) => {
                const videoId = parseInt(e.target.value);
                if (videoId) {
                    this.selectVideoForRenderer(videoId);
                }
            });
        }
    }

    setupUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.getElementById('upload-btn');

        if (!uploadArea || !fileInput || !uploadBtn) return;

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--accent-color)';
            uploadArea.style.background = 'rgba(59, 130, 246, 0.05)';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = 'var(--border-color)';
            uploadArea.style.background = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--border-color)';
            uploadArea.style.background = '';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        uploadBtn.addEventListener('click', () => {
            if (!uploadBtn.disabled) {
                this.uploadFile();
            }
        });
    }

    setupRefresh() {
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshAll());
        }
    }

    // Enhanced system status check
    async checkSystemStatus() {
        const connectionStatus = document.getElementById('connection-status');
        const apiStatus = document.getElementById('api-status');
        const gpuStatus = document.getElementById('gpu-status');
        const ffmpegStatus = document.getElementById('ffmpeg-status');

        try {
            // Check API connection
            const healthResponse = await fetch(`${this.apiUrl.replace('/api/v1', '')}/health`);
            if (healthResponse.ok) {
                connectionStatus.textContent = 'Connected';
                connectionStatus.className = 'status status-completed';

                apiStatus.textContent = 'Online';
                apiStatus.className = 'status status-completed';
            } else {
                throw new Error('API not available');
            }

            // Check renderer system
            const rendererResponse = await fetch(`${this.apiUrl}/renderer/system`);
            if (rendererResponse.ok) {
                const rendererData = await rendererResponse.json();

                ffmpegStatus.textContent = rendererData.ffmpeg_available ? 'Available' : 'Missing';
                ffmpegStatus.className = rendererData.ffmpeg_available ? 'status status-completed' : 'status status-error';

                // GPU status (simplified check)
                gpuStatus.textContent = 'Available';
                gpuStatus.className = 'status status-completed';
            }
        } catch (error) {
            console.error('System status check failed:', error);
            connectionStatus.textContent = 'Disconnected';
            connectionStatus.className = 'status status-error';

            apiStatus.textContent = 'Offline';
            apiStatus.className = 'status status-error';

            gpuStatus.textContent = 'Unknown';
            gpuStatus.className = 'status status-pending';

            ffmpegStatus.textContent = 'Unknown';
            ffmpegStatus.className = 'status status-pending';
        }
    }

    // Enhanced video loading
    async loadVideos() {
        const videosLoading = document.getElementById('videos-loading');
        const videosContainer = document.getElementById('videos-container');
        const noVideos = document.getElementById('no-videos');
        const videosGrid = document.getElementById('videos-grid');

        videosLoading.classList.remove('hidden');
        videosContainer.classList.add('hidden');
        noVideos.classList.add('hidden');

        try {
            const response = await fetch(`${this.apiUrl}/pipeline/videos`);
            if (!response.ok) {
                throw new Error('Failed to load videos');
            }

            const data = await response.json();
            this.videos = data.videos;

            videosLoading.classList.add('hidden');

            if (this.videos.length === 0) {
                noVideos.classList.remove('hidden');
            } else {
                videosContainer.classList.remove('hidden');
                this.renderVideoGrid();
            }

            this.updateVideoSelectors();

        } catch (error) {
            console.error('Failed to load videos:', error);
            videosLoading.classList.add('hidden');
            noVideos.classList.remove('hidden');
        }
    }

    renderVideoGrid() {
        const videosGrid = document.getElementById('videos-grid');
        if (!videosGrid) return;

        videosGrid.innerHTML = '';

        this.videos.forEach(video => {
            const videoCard = document.createElement('div');
            videoCard.className = 'video-card';
            if (this.selectedVideo && this.selectedVideo.id === video.id) {
                videoCard.classList.add('selected');
            }

            videoCard.innerHTML = `
                <div class="video-thumbnail">VIDEO</div>
                <div class="video-info">
                    <div class="video-title">${video.filename}</div>
                    <div class="video-meta">
                        ${this.formatFileSize(video.file_size)} • ${this.formatDate(video.created_at)}
                    </div>
                    <div class="video-status">
                        <span class="status ${this.getProcessingStatus(video).class}">${this.getProcessingStatus(video).text}</span>
                        ${video.pipeline_status?.shots_count ? `<span class="text-small text-muted">${video.pipeline_status.shots_count} shots</span>` : ''}
                    </div>
                    <div class="mt-1 flex gap-1">
                        <button class="btn btn-primary btn-small" onclick="window.app.selectVideo(${JSON.stringify(video).replace(/"/g, '&quot;')})">Select</button>
                        <button class="btn btn-danger btn-small" onclick="window.app.deleteVideo(${video.id}, event)">Delete</button>
                    </div>
                </div>
            `;

            videoCard.addEventListener('click', () => {
                this.selectVideo(video);
            });

            videosGrid.appendChild(videoCard);
        });
    }

    getProcessingStatus(video) {
        const status = video.pipeline_status;
        if (!status) return { class: 'status-pending', text: 'Not Processed' };

        if (status.shots_count && status.transcription_complete && status.analysis_complete) {
            return { class: 'status-completed', text: 'Complete' };
        } else if (status.processing) {
            return { class: 'status-processing', text: 'Processing' };
        } else if (status.shots_count) {
            return { class: 'status-processing', text: 'Partial' };
        }

        return { class: 'status-pending', text: 'Ready' };
    }

    selectVideo(video) {
        this.selectedVideo = video;
        this.updateSelectedVideoInfo();
        this.updateVideoSelectors();
        this.renderVideoGrid(); // Re-render to update selection
    }

    updateSelectedVideoInfo() {
        const selectedVideoInfo = document.getElementById('selected-video-info');
        if (!selectedVideoInfo) return;

        if (this.selectedVideo) {
            const status = this.getProcessingStatus(this.selectedVideo);
            selectedVideoInfo.innerHTML = `
                <div class="mb-1">
                    <strong>${this.selectedVideo.filename}</strong>
                </div>
                <div class="text-small text-muted mb-1">
                    ${this.formatFileSize(this.selectedVideo.file_size)}
                </div>
                <div class="text-small">
                    <span class="status ${status.class}">${status.text}</span>
                </div>
            `;
        } else {
            selectedVideoInfo.innerHTML = '<div class="text-center text-muted">No video selected</div>';
        }
    }

    updateVideoSelectors() {
        // Update all video selectors
        const selectors = [
            'pipeline-video-select',
            'editor-video-select',
            'renderer-video-select'
        ];

        selectors.forEach(selectorId => {
            const selector = document.getElementById(selectorId);
            if (selector) {
                selector.innerHTML = '<option value="">Select a video...</option>';
                this.videos.forEach(video => {
                    const option = document.createElement('option');
                    option.value = video.id;
                    option.textContent = video.filename;
                    if (this.selectedVideo && this.selectedVideo.id === video.id) {
                        option.selected = true;
                    }
                    selector.appendChild(option);
                });
            }
        });
    }

    // Pipeline functionality
    updatePipelineVideoSelector() {
        const pipelineVideoSelector = document.getElementById('pipeline-video-selector');
        const noVideoSelected = document.getElementById('no-video-selected');
        const pipelineSteps = document.getElementById('pipeline-steps');

        if (this.videos.length > 0) {
            pipelineVideoSelector.classList.remove('hidden');
        }

        if (this.selectedVideo) {
            noVideoSelected.classList.add('hidden');
            pipelineSteps.classList.remove('hidden');
            this.updatePipelineStatus();
        } else {
            noVideoSelected.classList.remove('hidden');
            pipelineSteps.classList.add('hidden');
        }
    }

    selectVideoForPipeline(videoId) {
        const video = this.videos.find(v => v.id === videoId);
        if (video) {
            this.selectVideo(video);
            this.updatePipelineVideoSelector();
        }
    }

    async updatePipelineStatus() {
        if (!this.selectedVideo) return;

        try {
            const response = await fetch(`${this.apiUrl}/pipeline/videos/${this.selectedVideo.id}/status`);
            if (response.ok) {
                const status = await response.json();

                // Shot Detection Status
                if (status.shots_count && status.shots_count > 0) {
                    this.updateStepStatus('shot-detection', 'completed', `${status.shots_count} shots detected`);
                } else {
                    this.updateStepStatus('shot-detection', 'pending', 'Ready to start');
                }

                // Transcription Status
                if (status.transcription_complete === true) {
                    this.updateStepStatus('transcription', 'completed', 'Completed');
                } else {
                    this.updateStepStatus('transcription', 'pending', 'Ready to start');
                }

                // Analysis Status
                if (status.analysis_complete === true) {
                    this.updateStepStatus('analysis', 'completed', 'Completed');
                } else {
                    this.updateStepStatus('analysis', 'pending', 'Ready to start');
                }
            }
        } catch (error) {
            console.error('Failed to update pipeline status:', error);
        }
    }

    updateStepStatus(stepName, status, text) {
        const statusElement = document.getElementById(`${stepName}-status`);
        const stepElement = document.getElementById(`step-${stepName}`);

        if (statusElement) {
            statusElement.textContent = text;
            statusElement.className = `status status-${status}`;
        }

        if (stepElement) {
            stepElement.className = `pipeline-step ${status}`;
        }
    }

    updateShotThreshold() {
        const method = document.getElementById('shot-method').value;
        const thresholdInput = document.getElementById('shot-threshold');
        const thresholdHelp = thresholdInput.nextElementSibling;

        if (method === 'pyscene') {
            thresholdInput.value = '30.0';
            thresholdInput.min = '0';
            thresholdInput.max = '100';
            thresholdInput.step = '1';
            thresholdHelp.textContent = 'Lower = more sensitive (more scene changes)';
        } else {
            thresholdInput.value = '0.5';
            thresholdInput.min = '0';
            thresholdInput.max = '1';
            thresholdInput.step = '0.1';
            thresholdHelp.textContent = 'Lower = more sensitive (more shots)';
        }
    }

    async startShotDetection() {
        if (!this.selectedVideo) return;

        const button = document.getElementById('start-shot-detection');
        const method = document.getElementById('shot-method').value;
        const threshold = parseFloat(document.getElementById('shot-threshold').value);

        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Processing...';
        this.updateStepStatus('shot-detection', 'processing', 'Processing...');

        try {
            const url = `${this.apiUrl}/pipeline/videos/${this.selectedVideo.id}/process?step=shot_detection&shot_method=${method}&shot_threshold=${threshold}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Shot detection failed');
            }

            const result = await response.json();
            this.showAlert('Shot detection completed successfully!', 'success');
            this.updatePipelineStatus();
            this.loadVideos(); // Refresh video data

        } catch (error) {
            console.error('Shot detection failed:', error);
            this.showAlert('Shot detection failed: ' + error.message, 'error');
            this.updateStepStatus('shot-detection', 'error', 'Failed');
        } finally {
            button.disabled = false;
            button.textContent = 'Start Shot Detection';
        }
    }

    async startTranscription() {
        if (!this.selectedVideo) return;

        const button = document.getElementById('start-transcription');
        const mode = document.querySelector('input[name="transcription-mode"]:checked').value;

        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Processing...';
        this.updateStepStatus('transcription', 'processing', 'Processing...');

        try {
            const url = `${this.apiUrl}/pipeline/videos/${this.selectedVideo.id}/process?step=transcription&per_shot=${mode === 'per-shot'}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Transcription failed');
            }

            this.showAlert('Transcription completed successfully!', 'success');
            this.updatePipelineStatus();
            this.loadVideos();

        } catch (error) {
            console.error('Transcription failed:', error);
            this.showAlert('Transcription failed: ' + error.message, 'error');
            this.updateStepStatus('transcription', 'error', 'Failed');
        } finally {
            button.disabled = false;
            button.textContent = 'Start Transcription';
        }
    }

    async startAnalysis() {
        if (!this.selectedVideo) return;

        const button = document.getElementById('start-analysis');
        const contentType = document.getElementById('content-type').value;

        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Processing...';
        this.updateStepStatus('analysis', 'processing', 'Processing...');

        try {
            const url = `${this.apiUrl}/pipeline/videos/${this.selectedVideo.id}/process?step=content_analysis&content_type=${contentType}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }

            this.showAlert('Content analysis completed successfully!', 'success');
            this.updatePipelineStatus();
            this.loadVideos();

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showAlert('Analysis failed: ' + error.message, 'error');
            this.updateStepStatus('analysis', 'error', 'Failed');
        } finally {
            button.disabled = false;
            button.textContent = 'Start Analysis';
        }
    }

    async runFullPipeline() {
        if (!this.selectedVideo) return;

        const button = document.getElementById('run-full-pipeline');
        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Running Pipeline...';

        try {
            // Run shot detection
            await this.startShotDetection();
            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for completion

            // Run transcription
            await this.startTranscription();
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Run analysis
            await this.startAnalysis();

            this.showAlert('Full pipeline completed successfully!', 'success');
        } catch (error) {
            this.showAlert('Pipeline failed: ' + error.message, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Run Complete Pipeline';
        }
    }

    async clearData(type) {
        if (!this.selectedVideo) return;

        if (!confirm(`Are you sure you want to clear ${type} data? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/pipeline/videos/${this.selectedVideo.id}/data`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type })
            });

            if (!response.ok) {
                throw new Error(`Failed to clear ${type}`);
            }

            this.showAlert(`${type} data cleared successfully!`, 'success');
            this.updatePipelineStatus();
            this.loadVideos();

        } catch (error) {
            console.error(`Failed to clear ${type}:`, error);
            this.showAlert(`Failed to clear ${type}: ` + error.message, 'error');
        }
    }

    async clearAllData() {
        if (!this.selectedVideo) return;

        if (!confirm('Are you sure you want to clear ALL processing data? This action cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/pipeline/videos/${this.selectedVideo.id}/data`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Failed to clear all data');
            }

            this.showAlert('All processing data cleared successfully!', 'success');
            this.updatePipelineStatus();
            this.loadVideos();

        } catch (error) {
            console.error('Failed to clear all data:', error);
            this.showAlert('Failed to clear data: ' + error.message, 'error');
        }
    }

    async getVideoStatus() {
        if (!this.selectedVideo) return;

        try {
            const response = await fetch(`${this.apiUrl}/pipeline/videos/${this.selectedVideo.id}/status`);
            if (response.ok) {
                const status = await response.json();
                this.showAlert('Status refreshed', 'info');
                this.updatePipelineStatus();
            }
        } catch (error) {
            console.error('Failed to get video status:', error);
            this.showAlert('Failed to refresh status', 'error');
        }
    }

    // Editor functionality
    updateEditorVideoSelector() {
        const editorVideoSelector = document.getElementById('editor-video-selector');
        const editorNoVideo = document.getElementById('editor-no-video');
        const editorContent = document.getElementById('editor-content');

        if (this.videos.length > 0) {
            editorVideoSelector.classList.remove('hidden');
        }

        if (this.selectedVideo) {
            editorNoVideo.classList.add('hidden');
            editorContent.classList.remove('hidden');
            this.loadEditorData();
        } else {
            editorNoVideo.classList.remove('hidden');
            editorContent.classList.add('hidden');
        }
    }

    selectVideoForEditor(videoId) {
        const video = this.videos.find(v => v.id === videoId);
        if (video) {
            this.selectVideo(video);
            this.updateEditorVideoSelector();
        }
    }

    async loadEditorData() {
        if (!this.selectedVideo) return;

        try {
            // Load timeline if available
            const timelineResponse = await fetch(`${this.apiUrl}/editor/${this.selectedVideo.id}/timeline`);
            const timelineContainer = document.getElementById('timeline-container');

            if (timelineResponse.ok) {
                const timeline = await timelineResponse.json();
                this.timeline = timeline.timeline;
                this.displayTimeline(timeline);
            } else {
                timelineContainer.innerHTML = '<p class="text-muted">No timeline available. Generate content first.</p>';
            }

            // Load content analysis if available
            this.loadContentAnalysis();
        } catch (error) {
            console.error('Failed to load editor data:', error);
        }
    }

    displayTimeline(timelineData) {
        const timelineContainer = document.getElementById('timeline-container');
        if (!timelineData || !timelineData.timeline) {
            timelineContainer.innerHTML = '<p class="text-muted">No timeline available. Generate content first.</p>';
            return;
        }

        const timeline = timelineData.timeline;
        let timelineHtml = '<div class="timeline-display">';

        timelineHtml += `
            <div class="timeline-header mb-2">
                <h4>Timeline Preview</h4>
                <div class="text-small text-muted">
                    Duration: ${Math.round(timeline.total_duration)}s •
                    ${timeline.scenes?.length || timeline.items?.length || 0} segments
                </div>
            </div>
        `;

        // Display timeline items
        const items = timeline.items || timeline.scenes || [];
        if (items.length > 0) {
            timelineHtml += '<div class="timeline-items">';
            items.forEach((item, index) => {
                const startTime = Math.round(item.start_time || 0);
                const endTime = Math.round(item.end_time || item.start_time + 5);
                const duration = endTime - startTime;

                timelineHtml += `
                    <div class="timeline-item" style="border: 1px solid var(--border-color); padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                        <div class="flex flex-between items-center mb-1">
                            <span class="text-small font-weight-bold">Segment ${index + 1}</span>
                            <span class="text-small text-muted">${startTime}s - ${endTime}s (${duration}s)</span>
                        </div>
                        <div class="text-small">
                            ${item.summary || item.highlight_reason || item.scene_type || 'Content segment'}
                        </div>
                        ${item.engagement_score ? `<div class="text-small text-muted">Engagement: ${Math.round(item.engagement_score * 10)/10}/10</div>` : ''}
                    </div>
                `;
            });
            timelineHtml += '</div>';
        }

        timelineHtml += '</div>';
        timelineContainer.innerHTML = timelineHtml;
    }

    async loadContentAnalysis() {
        if (!this.selectedVideo || !this.contentData) return;

        try {
            const response = await fetch(`${this.apiUrl}/editor/${this.selectedVideo.id}/content-potential`);
            if (response.ok) {
                const analysis = await response.json();
                this.displayContentAnalysis(analysis);
            }
        } catch (error) {
            console.error('Failed to load content analysis:', error);
        }
    }

    displayContentAnalysis(analysis) {
        const contentAnalysisEl = document.getElementById('content-analysis');
        if (!analysis) {
            contentAnalysisEl.innerHTML = '<p class="text-muted">Generate content to see analysis</p>';
            return;
        }

        console.log('Content analysis data:', analysis);

        const html = `
            <div class="analysis-metrics">
                <div class="metric mb-1">
                    <div class="flex flex-between">
                        <span class="text-small">Total Scenes</span>
                        <span class="text-small font-weight-bold">${analysis.total_scenes || analysis.total_clips || 0}</span>
                    </div>
                </div>
                <div class="metric mb-1">
                    <div class="flex flex-between">
                        <span class="text-small">Avg Engagement</span>
                        <span class="text-small font-weight-bold">${Math.round((analysis.avg_engagement || analysis.average_score || 0) * 10)/10}/10</span>
                    </div>
                </div>
                <div class="metric mb-1">
                    <div class="flex flex-between">
                        <span class="text-small">Viral Potential</span>
                        <span class="text-small font-weight-bold">${Math.round((analysis.avg_viral_potential || 0) * 10)/10}/10</span>
                    </div>
                </div>
                <div class="metric mb-1">
                    <div class="flex flex-between">
                        <span class="text-small">High Quality</span>
                        <span class="text-small font-weight-bold">${analysis.high_quality_clips || 0}</span>
                    </div>
                </div>
                <div class="metric mb-1">
                    <div class="flex flex-between">
                        <span class="text-small">Viral Potential</span>
                        <span class="text-small font-weight-bold">${Math.round((analysis.viral_potential || 0) * 10)/10}/10</span>
                    </div>
                </div>
                <div class="metric">
                    <div class="flex flex-between">
                        <span class="text-small">Content Density</span>
                        <span class="text-small font-weight-bold">${Math.round((analysis.content_density || 0) * 100)}%</span>
                    </div>
                </div>
                ${analysis.scene_types ? `
                    <div class="metric mt-1">
                        <div class="text-small mb-1">Scene Types:</div>
                        <div class="text-tiny">
                            ${Object.entries(analysis.scene_types).map(([type, count]) =>
                                `<span class="badge">${type}: ${count}</span>`
                            ).join(' ')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        contentAnalysisEl.innerHTML = html;
    }

    displayRecommendations(recommendations) {
        const recommendationsEl = document.getElementById('ai-recommendations');
        if (!recommendations || !recommendations.length) {
            recommendationsEl.innerHTML = '<p class="text-muted">Generate content to see recommendations</p>';
            return;
        }

        let html = '<div class="recommendations-list">';
        recommendations.forEach((rec, index) => {
            html += `
                <div class="recommendation-item mb-1 p-2" style="border-left: 3px solid var(--accent-color); background: var(--secondary-bg);">
                    <div class="text-small">${rec}</div>
                </div>
            `;
        });
        html += '</div>';

        recommendationsEl.innerHTML = html;
    }

    async generateContent() {
        if (!this.selectedVideo) return;

        const button = document.getElementById('generate-content');
        const platforms = Array.from(document.querySelectorAll('input[name="editor-platforms"]:checked')).map(cb => cb.value);
        const objective = document.getElementById('editor-objective').value || 'Create engaging social media content';

        if (platforms.length === 0) {
            this.showAlert('Please select at least one target platform', 'warning');
            return;
        }

        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Generating...';

        try {
            const response = await fetch(`${this.apiUrl}/editor/${this.selectedVideo.id}/edit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_brief: objective,
                    content_type: 'interview',
                    target_platform: platforms[0] // Use first platform for now
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Content generation failed');
            }

            const data = await response.json();
            this.contentData = data;

            this.showAlert('Content generated successfully!', 'success');
            this.loadEditorData(); // Refresh editor data

            // Display analysis and recommendations
            if (data.content_analysis) {
                this.displayContentAnalysis(data.content_analysis);
            }
            if (data.recommendations) {
                this.displayRecommendations(data.recommendations);
            }

        } catch (error) {
            console.error('Content generation failed:', error);
            this.showAlert('Content generation failed: ' + error.message, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Generate Content';
        }
    }

    async regenerateContent() {
        if (!this.selectedVideo) return;

        const button = document.getElementById('regenerate-content');
        const platforms = Array.from(document.querySelectorAll('input[name="editor-platforms"]:checked')).map(cb => cb.value);
        const objective = document.getElementById('editor-objective').value || 'Create engaging social media content';

        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Regenerating...';

        try {
            const response = await fetch(`${this.apiUrl}/editor/${this.selectedVideo.id}/regenerate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_brief: objective,
                    content_type: 'interview',
                    target_platform: platforms[0]
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Content regeneration failed');
            }

            const data = await response.json();
            this.contentData = data;
            this.showAlert('Content regenerated successfully!', 'success');
            this.loadEditorData(); // Refresh editor data

            // Display analysis and recommendations
            if (data.content_analysis) {
                this.displayContentAnalysis(data.content_analysis);
            }
            if (data.recommendations) {
                this.displayRecommendations(data.recommendations);
            }

        } catch (error) {
            console.error('Content regeneration failed:', error);
            this.showAlert('Content regeneration failed: ' + error.message, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Regenerate';
        }
    }

    // Renderer functionality
    updateRendererVideoSelector() {
        const rendererVideoSelector = document.getElementById('renderer-video-selector');
        const rendererNoVideo = document.getElementById('renderer-no-video');
        const rendererContent = document.getElementById('renderer-content');

        if (this.videos.length > 0) {
            rendererVideoSelector.classList.remove('hidden');
        }

        if (this.selectedVideo) {
            rendererNoVideo.classList.add('hidden');
            rendererContent.classList.remove('hidden');
            this.loadRendererData();
        } else {
            rendererNoVideo.classList.remove('hidden');
            rendererContent.classList.add('hidden');
        }
    }

    selectVideoForRenderer(videoId) {
        const video = this.videos.find(v => v.id === videoId);
        if (video) {
            this.selectVideo(video);
            this.updateRendererVideoSelector();
        }
    }

    async loadPlatformPresets() {
        try {
            const response = await fetch(`${this.apiUrl}/editor/platforms`);
            if (!response.ok) return;

            const data = await response.json();
            this.platformPresets = data.platforms;
            const presetsContainer = document.getElementById('platform-presets');

            if (presetsContainer) {
                presetsContainer.innerHTML = '';

                Object.entries(data.platforms).forEach(([key, platform]) => {
                    const preset = document.createElement('div');
                    preset.className = 'card platform-preset';
                    preset.style.cssText = 'margin: 0; padding: 15px; cursor: pointer; transition: var(--transition);';
                    preset.dataset.platform = key;

                    preset.innerHTML = `
                        <h4>${key.replace('_', ' ').toUpperCase()}</h4>
                        <div class="text-small text-muted">${platform.ratio} • ${platform.max_duration}s max</div>
                        <div class="text-small text-muted">${platform.format}</div>
                    `;

                    preset.addEventListener('click', () => {
                        document.querySelectorAll('.platform-preset').forEach(p => {
                            p.classList.remove('selected');
                            p.style.borderColor = 'var(--border-color)';
                        });
                        preset.classList.add('selected');
                        preset.style.borderColor = 'var(--accent-color)';
                        preset.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.1)';
                    });

                    presetsContainer.appendChild(preset);
                });
            }
        } catch (error) {
            console.error('Failed to load platform presets:', error);
        }
    }

    async loadRendererData() {
        try {
            // Always load render queue for all videos
            const jobsResponse = await fetch(`${this.apiUrl}/renderer/jobs`);
            if (jobsResponse.ok) {
                const jobs = await jobsResponse.json();
                this.updateRenderQueue(jobs.jobs || []);
            } else {
                this.updateRenderQueue([]);
            }

            // Load available outputs
            const outputsResponse = await fetch(`${this.apiUrl}/renderer/outputs`);
            if (outputsResponse.ok) {
                const outputs = await outputsResponse.json();
                this.updateDownloadCenter(outputs.files || []);
            } else {
                this.updateDownloadCenter([]);
            }
        } catch (error) {
            console.error('Failed to load renderer data:', error);
            this.updateRenderQueue([]);
            this.updateDownloadCenter([]);
        }
    }

    startRendererDataRefresh() {
        // Clear any existing refresh timer
        if (this.rendererRefreshTimer) {
            clearInterval(this.rendererRefreshTimer);
        }

        // Refresh every 5 seconds when on renderer tab
        this.rendererRefreshTimer = setInterval(() => {
            if (this.currentTab === 'renderer') {
                this.loadRendererData();
            } else {
                // Stop refreshing if not on renderer tab
                clearInterval(this.rendererRefreshTimer);
                this.rendererRefreshTimer = null;
            }
        }, 5000);
    }

    updateRenderQueue(jobs) {
        const queueList = document.getElementById('render-queue-list');
        if (!queueList) return;

        if (!jobs || jobs.length === 0) {
            queueList.innerHTML = '<p class="text-muted">No render jobs in queue</p>';
            return;
        }

        queueList.innerHTML = '';
        jobs.forEach(job => {
            const jobElement = document.createElement('div');
            jobElement.className = 'render-job';
            jobElement.style.cssText = 'border: 1px solid var(--border-color); padding: 10px; margin-bottom: 10px; border-radius: 4px;';

            jobElement.innerHTML = `
                <div class="flex flex-between items-center mb-1">
                    <span class="text-small font-weight-bold">${job.job_id}</span>
                    <span class="status status-${job.status}">${job.status}</span>
                </div>
                <div class="text-small text-muted">
                    ${job.output_filename || 'output.mp4'} • ${job.settings?.quality || 'high'} quality
                </div>
                ${job.progress !== undefined ? `
                    <div class="progress mt-1" style="height: 4px;">
                        <div class="progress-bar" style="width: ${job.progress}%;"></div>
                    </div>
                ` : ''}
            `;

            queueList.appendChild(jobElement);
        });
    }

    updateDownloadCenter(outputs) {
        const downloadList = document.getElementById('download-list');
        if (!downloadList) return;

        if (!outputs || outputs.length === 0) {
            downloadList.innerHTML = '<p class="text-muted">No completed renders</p>';
            return;
        }

        downloadList.innerHTML = '';
        outputs.forEach(output => {
            const outputElement = document.createElement('div');
            outputElement.className = 'download-item';
            outputElement.style.cssText = 'border: 1px solid var(--border-color); padding: 10px; margin-bottom: 10px; border-radius: 4px;';

            outputElement.innerHTML = `
                <div class="flex flex-between items-center mb-1">
                    <span class="text-small font-weight-bold">${output.filename}</span>
                    <span class="text-small text-muted">${output.size_mb.toFixed(1)} MB</span>
                </div>
                <div class="text-small text-muted mb-1">
                    Created: ${this.formatDate(output.created_at)}
                </div>
                <button class="btn btn-primary btn-small" onclick="window.app.downloadFile('${output.filename}')">
                    Download
                </button>
            `;

            downloadList.appendChild(outputElement);
        });
    }

    async startRender() {
        if (!this.selectedVideo) {
            this.showAlert('Please select a video first', 'warning');
            return;
        }

        if (!this.contentData && !this.timeline) {
            this.showAlert('Please generate content first', 'warning');
            return;
        }

        const selectedPreset = document.querySelector('.platform-preset.selected');
        const quality = document.getElementById('render-quality').value;
        const useGpu = document.getElementById('gpu-acceleration').checked;
        const audioEnhancement = document.getElementById('audio-enhancement').checked;

        const button = document.getElementById('start-render');
        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Starting Render...';

        try {
            // Map platform to video format
            let videoFormat = 'portrait';
            if (selectedPreset) {
                const platform = selectedPreset.dataset.platform;
                const preset = this.platformPresets?.[platform];
                if (preset?.format) {
                    videoFormat = preset.format;
                }
            }

            const renderData = {
                video_format: videoFormat,
                quality: quality,
                use_gpu: useGpu,
                audio_enhancement: audioEnhancement
            };

            const response = await fetch(`${this.apiUrl}/renderer/${this.selectedVideo.id}/render`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(renderData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Render failed to start');
            }

            const result = await response.json();
            this.showAlert('Render started successfully!', 'success');
            this.loadRendererData(); // Refresh render data
            this.startRenderProgressMonitoring(result.job_id);

        } catch (error) {
            console.error('Render failed:', error);
            this.showAlert('Render failed: ' + error.message, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Start Render';
        }
    }

    startRenderProgressMonitoring(jobId) {
        const progressInfo = document.getElementById('render-progress-info');
        const progressBarContainer = document.getElementById('render-progress-bar');
        const progressBar = document.getElementById('render-progress');

        progressInfo.innerHTML = `
            <div class="text-small">
                <span class="spinner"></span> Rendering in progress...
            </div>
            <div class="text-small text-muted">Job ID: ${jobId}</div>
        `;
        progressBarContainer.classList.remove('hidden');

        const checkProgress = async () => {
            try {
                const response = await fetch(`${this.apiUrl}/renderer/jobs/${jobId}/status`);
                if (response.ok) {
                    const status = await response.json();

                    if (status.progress !== undefined) {
                        progressBar.style.width = `${status.progress}%`;
                    }

                    if (status.status === 'completed') {
                        progressInfo.innerHTML = '<div class="text-small text-success">Render completed successfully!</div>';
                        progressBarContainer.classList.add('hidden');
                        await this.loadRendererData(); // Refresh render queue and download center
                        this.showAlert('Render completed successfully!', 'success');
                        return; // Stop monitoring
                    } else if (status.status === 'failed') {
                        progressInfo.innerHTML = '<div class="text-small text-error">Render failed</div>';
                        progressBarContainer.classList.add('hidden');
                        await this.loadRendererData(); // Refresh render queue
                        this.showAlert('Render failed', 'error');
                        return; // Stop monitoring
                    }

                    // Continue monitoring
                    setTimeout(checkProgress, 2000);
                }
            } catch (error) {
                console.error('Failed to check render progress:', error);
            }
        };

        checkProgress();
    }

    async addToRenderQueue() {
        if (!this.selectedVideo) {
            this.showAlert('Please select a video first', 'warning');
            return;
        }

        const selectedPreset = document.querySelector('.platform-preset.selected');
        const quality = document.getElementById('render-quality').value;

        try {
            const queueData = {
                video_id: this.selectedVideo.id,
                platform: selectedPreset ? selectedPreset.dataset.platform : 'youtube_shorts',
                quality: quality
            };

            const response = await fetch(`${this.apiUrl}/renderer/queue`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(queueData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to add to queue');
            }

            this.showAlert('Added to render queue successfully!', 'success');
            this.loadRendererData();

        } catch (error) {
            console.error('Failed to add to queue:', error);
            this.showAlert('Failed to add to queue: ' + error.message, 'error');
        }
    }

    async downloadFile(filename) {
        try {
            const response = await fetch(`${this.apiUrl}/renderer/outputs/${filename}/download`);
            if (!response.ok) {
                throw new Error('Download failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            this.showAlert('Download started!', 'success');
        } catch (error) {
            console.error('Download failed:', error);
            this.showAlert('Download failed: ' + error.message, 'error');
        }
    }

    // Upload functionality
    handleFileSelect(file) {
        const uploadBtn = document.getElementById('upload-btn');
        const uploadArea = document.getElementById('upload-area');

        if (file && file.type.startsWith('video/')) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = `Upload ${file.name}`;
            uploadArea.innerHTML = `
                <div style="font-size: 48px; color: var(--success-color); margin-bottom: 10px;">READY</div>
                <h3>Ready to upload</h3>
                <p class="text-muted">${file.name} (${this.formatFileSize(file.size)})</p>
            `;
            this.selectedFile = file;
        } else {
            this.showAlert('Please select a valid video file', 'error');
        }
    }

    async uploadFile() {
        if (!this.selectedFile) return;

        const uploadBtn = document.getElementById('upload-btn');
        const uploadProgress = document.getElementById('upload-progress');
        const uploadProgressBar = document.getElementById('upload-progress-bar');
        const uploadPercentage = document.getElementById('upload-percentage');

        uploadBtn.disabled = true;
        uploadProgress.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', this.selectedFile);

        try {
            const response = await fetch(`${this.apiUrl}/upload/`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();
            this.showAlert('Video uploaded successfully!', 'success');

            // Reset upload UI
            uploadProgress.classList.add('hidden');
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Upload Video';
            document.getElementById('upload-area').innerHTML = `
                <div style="font-size: 48px; color: var(--text-secondary); margin-bottom: 10px;">UPLOAD</div>
                <h3>Drop your video here</h3>
                <p class="text-muted">or click to select files</p>
                <p class="text-small text-muted mt-1">Supports MP4, MOV, AVI • Max 2GB</p>
            `;

            // Refresh video list
            this.loadVideos();

        } catch (error) {
            console.error('Upload failed:', error);
            this.showAlert('Upload failed: ' + error.message, 'error');
            uploadProgress.classList.add('hidden');
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Try Again';
        }
    }

    // Status polling
    startStatusPolling() {
        // Poll for status updates every 5 seconds
        setInterval(() => {
            if (this.selectedVideo) {
                this.updatePipelineStatus();
            }
        }, 5000);
    }

    async refreshAll() {
        const refreshBtn = document.getElementById('refresh-btn');
        const refreshSpinner = document.getElementById('refresh-spinner');

        refreshBtn.disabled = true;
        refreshSpinner.classList.remove('hidden');

        try {
            await Promise.all([
                this.checkSystemStatus(),
                this.loadVideos(),
                this.selectedVideo ? this.updatePipelineStatus() : Promise.resolve(),
                this.selectedVideo ? this.loadEditorData() : Promise.resolve(),
                this.selectedVideo ? this.loadRendererData() : Promise.resolve()
            ]);

            this.showAlert('All data refreshed successfully!', 'success');
        } catch (error) {
            console.error('Refresh failed:', error);
            this.showAlert('Refresh failed: ' + error.message, 'error');
        } finally {
            refreshBtn.disabled = false;
            refreshSpinner.classList.add('hidden');
        }
    }

    async deleteVideo(videoId, event) {
        event.stopPropagation(); // Prevent video selection when clicking delete

        if (!confirm('Are you sure you want to delete this video? This action cannot be undone and will remove all associated data.')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/upload/${videoId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete video');
            }

            this.showAlert('Video deleted successfully', 'success');

            // Clear selection if deleted video was selected
            if (this.selectedVideo && this.selectedVideo.id === videoId) {
                this.selectedVideo = null;
                this.updateSelectedVideoInfo();
                this.updateVideoSelectors();
            }

            // Reload video list
            this.loadVideos();

        } catch (error) {
            console.error('Failed to delete video:', error);
            this.showAlert('Failed to delete video: ' + error.message, 'error');
        }
    }

    // Utility functions
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDate(dateString) {
        return new Date(dateString).toLocaleDateString();
    }

    showAlert(message, type = 'info') {
        // Create a temporary alert
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 1000; max-width: 400px; box-shadow: var(--shadow-lg);';
        alert.innerHTML = `
            ${message}
            <button class="close-btn" onclick="this.parentElement.remove()" style="position: absolute; top: 5px; right: 10px; background: none; border: none; font-size: 18px; cursor: pointer;">×</button>
        `;

        document.body.appendChild(alert);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }
}

// Global functions for inline event handlers
function toggleStep(stepName) {
    const content = document.getElementById(`${stepName}-content`);
    const header = content.previousElementSibling;
    const arrow = header.querySelector('span:last-child');

    if (content.classList.contains('show')) {
        content.classList.remove('show');
        arrow.textContent = '▼';
    } else {
        content.classList.add('show');
        arrow.textContent = '▲';
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AICoCreatorManualApp();
});