// Campus Network IDS Dashboard JavaScript

// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000', // Update with your actual backend URL
    REFRESH_INTERVAL: 10000, // 10 seconds
    CHART_COLORS: {
        LOW: '#26de81',
        MEDIUM: '#ffa726',
        HIGH: '#ff9800',
        CRITICAL: '#ff4757'
    }
};

// Global variables
let alertsData = [];
let autoRefreshEnabled = true;
let refreshInterval;
let severityChart;
let timelineChart;
let currentUser = null;
let authToken = null;

// DOM Elements
const elements = {
    totalAlerts: document.getElementById('total-alerts'),
    highSeverityAlerts: document.getElementById('high-severity-alerts'),
    criticalAlerts: document.getElementById('critical-alerts'),
    openAlerts: document.getElementById('open-alerts'),
    alertsTableBody: document.getElementById('alerts-table-body'),
    displayedAlertsCount: document.getElementById('displayed-alerts-count'),
    timestamp: document.getElementById('timestamp'),
    statusIndicator: document.getElementById('status-indicator'),
    loadingState: document.getElementById('loading-state'),
    noAlertsState: document.getElementById('no-alerts-state'),
    errorState: document.getElementById('error-state'),
    errorMessage: document.getElementById('error-message'),
    refreshBtn: document.getElementById('refresh-btn'),
    simulateAttackBtn: document.getElementById('simulate-attack-btn'),
    autoRefreshCheckbox: document.getElementById('auto-refresh'),
    toastContainer: document.getElementById('toast-container'),
    userInfo: document.getElementById('user-info'),
    userName: document.getElementById('user-name'),
    userRole: document.getElementById('user-role'),
    logoutBtn: document.getElementById('logout-btn')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    checkAuthentication();
});

// Check authentication before initializing
function checkAuthentication() {
    console.log('🔐 Checking authentication state...');
    
    // Listen for auth state changes
    AuthService.onAuthStateChanged(async (user) => {
        if (user && user.emailVerified) {
            try {
                // Get and store auth token
                authToken = await user.getIdToken();
                currentUser = user;
                
                // Verify with backend
                await verifyUserWithBackend();
                
                // Initialize dashboard
                initializeApp();
                setupEventListeners();
                initializeCharts();
                
                console.log('✅ Authentication successful, initializing dashboard');
                
            } catch (error) {
                console.error('❌ Authentication verification failed:', error);
                redirectToLogin();
            }
        } else {
            console.log('❌ User not authenticated or email not verified');
            redirectToLogin();
        }
    });
}

function redirectToLogin() {
    showToast('Authentication required. Redirecting to login...', 'warning');
    setTimeout(() => {
        window.location.href = 'login.html';
    }, 2000);
}

async function verifyUserWithBackend() {
    try {
        const response = await AuthService.apiRequest(`${CONFIG.API_BASE_URL}/api/auth/me`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Authentication failed`);
        }
        
        const userProfile = await response.json();
        updateUserInterface(userProfile);
        
        return userProfile;
        
    } catch (error) {
        console.error('Backend verification failed:', error);
        throw error;
    }
}

// Authentication and User Interface Functions
function updateUserInterface(userProfile) {
    // Show user info
    elements.userInfo.style.display = 'flex';
    elements.userName.textContent = userProfile.name || userProfile.email.split('@')[0];
    elements.userRole.textContent = userProfile.role;
    
    // Add role styling
    elements.userRole.className = `user-role ${userProfile.role}`;
    
    // Show/hide features based on role and permissions
    const isAdmin = userProfile.role === 'admin';
    const canCreateAlerts = userProfile.email_verified;
    
    // Show simulate attack button only for verified users
    if (canCreateAlerts) {
        elements.simulateAttackBtn.style.display = 'inline-flex';
    }
    
    // Store user permissions for later use
    window.userPermissions = {
        canViewAlerts: true,
        canCreateAlerts: canCreateAlerts,
        canResolveAlerts: isAdmin,
        isAdmin: isAdmin
    };
    
    console.log('👤 User interface updated:', {
        name: userProfile.name || userProfile.email,
        role: userProfile.role,
        permissions: window.userPermissions
    });
}

async function handleLogout() {
    try {
        showToast('Signing out...', 'info');
        
        // Clear local state
        currentUser = null;
        authToken = null;
        stopAutoRefresh();
        
        // Sign out from Firebase
        await AuthService.signOut();
        
        showToast('Signed out successfully', 'success');
        
        // Redirect to login
        setTimeout(() => {
            window.location.href = 'login.html';
        }, 1500);
        
    } catch (error) {
        console.error('Logout failed:', error);
        showToast('Failed to sign out', 'error');
    }
}

async function refreshAuthToken() {
    try {
        if (currentUser) {
            authToken = await currentUser.getIdToken(true); // Force refresh
            console.log('🔄 Auth token refreshed');
        }
    } catch (error) {
        console.error('Failed to refresh token:', error);
        // If token refresh fails, redirect to login
        redirectToLogin();
    }
}

// Initialize the application
function initializeApp() {
    console.log('🚀 Initializing Campus Network IDS Dashboard');
    loadAlerts();
    startAutoRefresh();
    updateTimestamp();
}

// Setup event listeners
function setupEventListeners() {
    elements.refreshBtn.addEventListener('click', () => {
        showToast('Refreshing alerts...', 'info');
        loadAlerts();
    });

    elements.simulateAttackBtn.addEventListener('click', simulateAttack);
    
    elements.logoutBtn.addEventListener('click', handleLogout);

    elements.autoRefreshCheckbox.addEventListener('change', function() {
        autoRefreshEnabled = this.checked;
        if (autoRefreshEnabled) {
            startAutoRefresh();
            showToast('Auto-refresh enabled', 'success');
        } else {
            stopAutoRefresh();
            showToast('Auto-refresh disabled', 'warning');
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'r':
                    e.preventDefault();
                    loadAlerts();
                    break;
                case 's':
                    e.preventDefault();
                    simulateAttack();
                    break;
                case 'l':
                    e.preventDefault();
                    handleLogout();
                    break;
            }
        }
    });
}

// Load alerts from API
async function loadAlerts() {
    try {
        showLoadingState();
        updateStatus('loading');
        
        const response = await AuthService.apiRequest(`${CONFIG.API_BASE_URL}/api/alerts`);

        if (!response.ok) {
            if (response.status === 401) {
                showToast('Session expired. Please sign in again.', 'warning');
                await AuthService.signOut();
                redirectToLogin();
                return;
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const alerts = await response.json();
        alertsData = alerts;
        
        await loadAlertStats();
        renderAlerts(alerts);
        updateCharts();
        updateStatus('online');
        updateTimestamp();
        hideLoadingState();
        
        console.log(`📊 Loaded ${alerts.length} alerts`);

    } catch (error) {
        console.error('❌ Failed to load alerts:', error);
        showErrorState(error.message);
        updateStatus('offline');
        showToast(`Failed to load alerts: ${error.message}`, 'error');
    }
}

// Load alert statistics
async function loadAlertStats() {
    try {
        const response = await AuthService.apiRequest(`${CONFIG.API_BASE_URL}/api/alerts/stats/summary`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const stats = await response.json();
        updateSummaryCards(stats);
        
    } catch (error) {
        console.error('❌ Failed to load alert stats:', error);
        // Use fallback calculation from alerts data
        calculateStatsFromAlerts();
    }
}

// Calculate stats from loaded alerts (fallback)
function calculateStatsFromAlerts() {
    const stats = {
        total_alerts: alertsData.length,
        high_severity_alerts: alertsData.filter(a => a.severity === 'HIGH').length,
        critical_alerts: alertsData.filter(a => a.severity === 'CRITICAL').length,
        open_alerts: alertsData.filter(a => a.status === 'OPEN').length
    };
    updateSummaryCards(stats);
}

// Update summary cards
function updateSummaryCards(stats) {
    elements.totalAlerts.textContent = stats.total_alerts || 0;
    elements.highSeverityAlerts.textContent = stats.high_severity_alerts || 0;
    elements.criticalAlerts.textContent = stats.critical_alerts || 0;
    elements.openAlerts.textContent = stats.open_alerts || 0;

    // Add animation effect
    [elements.totalAlerts, elements.highSeverityAlerts, elements.criticalAlerts, elements.openAlerts]
        .forEach(el => {
            el.style.transform = 'scale(1.1)';
            setTimeout(() => {
                el.style.transform = 'scale(1)';
            }, 200);
        });
}

// Render alerts in the table
function renderAlerts(alerts) {
    const tbody = elements.alertsTableBody;
    tbody.innerHTML = '';

    if (alerts.length === 0) {
        showNoAlertsState();
        return;
    }

    alerts.forEach(alert => {
        const row = createAlertRow(alert);
        tbody.appendChild(row);
    });

    elements.displayedAlertsCount.textContent = alerts.length;
    hideNoAlertsState();
}

// Create a table row for an alert
function createAlertRow(alert) {
    const row = document.createElement('tr');
    row.setAttribute('data-alert-id', alert.id);
    
    // Add severity-based row styling
    row.classList.add(`alert-row-${alert.severity.toLowerCase()}`);
    
    row.innerHTML = `
        <td><span class="ip-address">${escapeHtml(alert.source_ip)}</span></td>
        <td><span class="ip-address">${escapeHtml(alert.destination_ip)}</span></td>
        <td>${escapeHtml(alert.attack_type)}</td>
        <td><span class="severity-badge severity-${alert.severity}">${alert.severity}</span></td>
        <td>
            ${alert.anomaly_score !== null 
                ? `<span class="anomaly-score anomaly-${getAnomalyLevel(alert.anomaly_score)}">${alert.anomaly_score.toFixed(3)}</span>`
                : '<span class="anomaly-score">N/A</span>'
            }
        </td>
        <td>${formatTimestamp(alert.timestamp)}</td>
        <td><span class="status-badge status-${alert.status}">${alert.status}</span></td>
        <td>
            ${alert.status === 'OPEN' && window.userPermissions?.canResolveAlerts 
                ? `<button class="resolve-btn" onclick="resolveAlert('${alert.id}')">
                     <i class="fas fa-check"></i> Resolve
                   </button>`
                : alert.status === 'RESOLVED' 
                    ? '<span class="text-muted">Resolved</span>'
                    : '<span class="text-muted">Admin Only</span>'
            }
        </td>
    `;

    // Add click animation
    row.addEventListener('click', function() {
        row.style.background = 'rgba(0, 212, 170, 0.1)';
        setTimeout(() => {
            row.style.background = '';
        }, 300);
    });

    return row;
}

// Resolve an alert
async function resolveAlert(alertId) {
    // Check permissions
    if (!window.userPermissions?.canResolveAlerts) {
        showToast('Admin access required to resolve alerts', 'error');
        return;
    }
    
    try {
        const button = document.querySelector(`tr[data-alert-id="${alertId}"] .resolve-btn`);
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Resolving...';

        const response = await AuthService.apiRequest(`${CONFIG.API_BASE_URL}/api/alerts/${alertId}/resolve`, {
            method: 'PUT'
        });

        if (!response.ok) {
            if (response.status === 403) {
                showToast('Admin access required to resolve alerts', 'error');
                return;
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const resolvedAlert = await response.json();
        
        showToast('Alert resolved successfully!', 'success');
        
        // Update the local data
        const alertIndex = alertsData.findIndex(a => a.id === alertId);
        if (alertIndex !== -1) {
            alertsData[alertIndex].status = 'RESOLVED';
        }
        
        // Refresh the display
        loadAlerts();

    } catch (error) {
        console.error('❌ Failed to resolve alert:', error);
        showToast(`Failed to resolve alert: ${error.message}`, 'error');
        
        // Re-enable the button
        const button = document.querySelector(`tr[data-alert-id="${alertId}"] .resolve-btn`);
        if (button) {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-check"></i> Resolve';
        }
    }
}

// Simulate a network attack
async function simulateAttack() {
    // Check permissions
    if (!window.userPermissions?.canCreateAlerts) {
        showToast('Email verification required to simulate attacks', 'error');
        return;
    }
    
    try {
        elements.simulateAttackBtn.disabled = true;
        elements.simulateAttackBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Simulating...';

        const response = await AuthService.apiRequest(`${CONFIG.API_BASE_URL}/api/simulate-attack`, {
            method: 'POST'
        });

        if (!response.ok) {
            if (response.status === 403) {
                showToast('Email verification required to simulate attacks', 'error');
                return;
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        
        showToast('Attack simulation created successfully!', 'success');
        
        // Refresh alerts to show the new simulated attack
        setTimeout(() => {
            loadAlerts();
        }, 1000);

    } catch (error) {
        console.error('❌ Failed to simulate attack:', error);
        showToast(`Failed to simulate attack: ${error.message}`, 'error');
    } finally {
        elements.simulateAttackBtn.disabled = false;
        elements.simulateAttackBtn.innerHTML = '<i class="fas fa-bug"></i> Simulate Attack';
    }
}

// Initialize Chart.js charts
function initializeCharts() {
    // Severity Chart
    const severityCtx = document.getElementById('severityChart').getContext('2d');
    severityChart = new Chart(severityCtx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Medium', 'High', 'Critical'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: [
                    CONFIG.CHART_COLORS.LOW,
                    CONFIG.CHART_COLORS.MEDIUM,
                    CONFIG.CHART_COLORS.HIGH,
                    CONFIG.CHART_COLORS.CRITICAL
                ],
                borderWidth: 2,
                borderColor: '#252b3a'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#b4bac7',
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                }
            },
            elements: {
                arc: {
                    borderWidth: 2
                }
            }
        }
    });

    // Timeline Chart
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    timelineChart = new Chart(timelineCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Alerts',
                data: [],
                borderColor: CONFIG.CHART_COLORS.HIGH,
                backgroundColor: 'rgba(255, 152, 0, 0.1)',
                tension: 0.4,
                fill: true,
                pointBackgroundColor: CONFIG.CHART_COLORS.HIGH,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#b4bac7'
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#b4bac7'
                    },
                    grid: {
                        color: 'rgba(180, 186, 199, 0.1)'
                    }
                },
                y: {
                    ticks: {
                        color: '#b4bac7'
                    },
                    grid: {
                        color: 'rgba(180, 186, 199, 0.1)'
                    }
                }
            }
        }
    });
}

// Update charts with current data
function updateCharts() {
    updateSeverityChart();
    updateTimelineChart();
}

// Update severity chart
function updateSeverityChart() {
    const severityCounts = {
        LOW: alertsData.filter(a => a.severity === 'LOW').length,
        MEDIUM: alertsData.filter(a => a.severity === 'MEDIUM').length,
        HIGH: alertsData.filter(a => a.severity === 'HIGH').length,
        CRITICAL: alertsData.filter(a => a.severity === 'CRITICAL').length
    };

    severityChart.data.datasets[0].data = [
        severityCounts.LOW,
        severityCounts.MEDIUM,
        severityCounts.HIGH,
        severityCounts.CRITICAL
    ];

    severityChart.update('none');
}

// Update timeline chart (alerts in the last 24 hours by hour)
function updateTimelineChart() {
    const now = new Date();
    const hours = [];
    const alertCounts = [];

    // Generate last 24 hours
    for (let i = 23; i >= 0; i--) {
        const hour = new Date(now.getTime() - (i * 60 * 60 * 1000));
        hours.push(hour.getHours().toString().padStart(2, '0') + ':00');
        
        // Count alerts in this hour
        const hourStart = new Date(hour.getTime());
        hourStart.setMinutes(0, 0, 0);
        const hourEnd = new Date(hourStart.getTime() + 60 * 60 * 1000);
        
        const alertsInHour = alertsData.filter(alert => {
            const alertTime = new Date(alert.timestamp);
            return alertTime >= hourStart && alertTime < hourEnd;
        }).length;
        
        alertCounts.push(alertsInHour);
    }

    timelineChart.data.labels = hours;
    timelineChart.data.datasets[0].data = alertCounts;
    timelineChart.update('none');
}

// Auto-refresh functionality
function startAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    
    refreshInterval = setInterval(() => {
        if (autoRefreshEnabled) {
            loadAlerts();
        }
    }, CONFIG.REFRESH_INTERVAL);
}

function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

// UI State Management
function showLoadingState() {
    elements.loadingState.style.display = 'block';
    elements.noAlertsState.style.display = 'none';
    elements.errorState.style.display = 'none';
    document.querySelector('.table-container').style.display = 'none';
}

function hideLoadingState() {
    elements.loadingState.style.display = 'none';
    document.querySelector('.table-container').style.display = 'block';
}

function showNoAlertsState() {
    elements.noAlertsState.style.display = 'block';
    elements.errorState.style.display = 'none';
    document.querySelector('.table-container').style.display = 'none';
}

function hideNoAlertsState() {
    elements.noAlertsState.style.display = 'none';
}

function showErrorState(message) {
    elements.errorState.style.display = 'block';
    elements.noAlertsState.style.display = 'none';
    elements.loadingState.style.display = 'none';
    elements.errorMessage.textContent = message;
    document.querySelector('.table-container').style.display = 'none';
}

function updateStatus(status) {
    const indicator = elements.statusIndicator;
    const icon = indicator.querySelector('i');
    const text = indicator.querySelector('span');

    switch (status) {
        case 'online':
            indicator.className = 'status-indicator';
            indicator.style.background = 'rgba(38, 222, 129, 0.1)';
            indicator.style.borderColor = '#26de81';
            indicator.style.color = '#26de81';
            icon.className = 'fas fa-circle';
            text.textContent = 'Online';
            break;
        case 'loading':
            indicator.style.background = 'rgba(255, 167, 38, 0.1)';
            indicator.style.borderColor = '#ffa726';
            indicator.style.color = '#ffa726';
            icon.className = 'fas fa-sync-alt fa-spin';
            text.textContent = 'Loading...';
            break;
        case 'offline':
            indicator.style.background = 'rgba(255, 71, 87, 0.1)';
            indicator.style.borderColor = '#ff4757';
            indicator.style.color = '#ff4757';
            icon.className = 'fas fa-exclamation-triangle';
            text.textContent = 'Offline';
            break;
    }
}

function updateTimestamp() {
    elements.timestamp.textContent = new Date().toLocaleString();
}

// Toast notification system
function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = getToastIcon(type);
    toast.innerHTML = `
        <i class="${icon}"></i>
        <span>${escapeHtml(message)}</span>
    `;

    elements.toastContainer.appendChild(toast);

    // Auto remove
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (toast.parentNode) {
                elements.toastContainer.removeChild(toast);
            }
        }, 300);
    }, duration);

    // Click to dismiss
    toast.addEventListener('click', () => {
        if (toast.parentNode) {
            elements.toastContainer.removeChild(toast);
        }
    });
}

function getToastIcon(type) {
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };
    return icons[type] || icons.info;
}

// Utility functions
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));

    if (diffInMinutes < 1) {
        return 'Just now';
    } else if (diffInMinutes < 60) {
        return `${diffInMinutes}m ago`;
    } else if (diffInMinutes < 1440) {
        return `${Math.floor(diffInMinutes / 60)}h ago`;
    } else {
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
}

function getAnomalyLevel(score) {
    if (score > 0.8) return 'high';
    if (score > 0.6) return 'medium';
    return 'low';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export functions for global access
window.resolveAlert = resolveAlert;
window.loadAlerts = loadAlerts;

console.log('✅ Campus Network IDS Dashboard loaded successfully!');
