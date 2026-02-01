// Authentication JavaScript for Campus Network IDS

// DOM Elements
const elements = {
    // Forms
    loginForm: document.getElementById('login-form'),
    registerForm: document.getElementById('register-form'),
    resetForm: document.getElementById('reset-form'),
    
    // Tab buttons
    tabButtons: document.querySelectorAll('.tab-button'),
    
    // Form inputs
    loginEmail: document.getElementById('login-email'),
    loginPassword: document.getElementById('login-password'),
    registerEmail: document.getElementById('register-email'),
    registerPassword: document.getElementById('register-password'),
    confirmPassword: document.getElementById('confirm-password'),
    resetEmail: document.getElementById('reset-email'),
    
    // Buttons
    loginSubmit: document.getElementById('login-submit'),
    registerSubmit: document.getElementById('register-submit'),
    resetSubmit: document.getElementById('reset-submit'),
    forgotPasswordBtn: document.getElementById('forgot-password-btn'),
    backToLoginBtn: document.getElementById('back-to-login-btn'),
    
    // Password toggles
    passwordToggles: document.querySelectorAll('.password-toggle'),
    
    // Loading and toast
    authLoading: document.getElementById('auth-loading'),
    toastContainer: document.getElementById('toast-container')
};

// Initialize the authentication page
document.addEventListener('DOMContentLoaded', function() {
    initializeAuthPage();
});

function initializeAuthPage() {
    console.log('🔐 Initializing Campus Network IDS Authentication');
    
    // Check if user is already authenticated
    checkAuthState();
    
    // Setup event listeners
    setupEventListeners();
    
    // Setup form validation
    setupFormValidation();
    
    // Handle URL parameters
    handleUrlParameters();
}

function checkAuthState() {
    // Listen for auth state changes
    AuthService.onAuthStateChanged((user) => {
        if (user) {
            if (user.emailVerified) {
                // User is signed in and verified, redirect to dashboard
                showToast('Welcome back! Redirecting to dashboard...', 'success');
                setTimeout(() => {
                    window.location.href = 'index.html';
                }, 1500);
            } else {
                // User is signed in but not verified
                showToast('Please verify your email address to access the dashboard', 'warning');
                showEmailVerificationPrompt(user);
            }
        }
    });
}

function setupEventListeners() {
    // Tab switching
    elements.tabButtons.forEach(button => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });
    
    // Form submissions
    elements.loginForm?.addEventListener('submit', handleLogin);
    elements.registerForm?.addEventListener('submit', handleRegister);
    elements.resetForm?.addEventListener('submit', handlePasswordReset);
    
    // Navigation buttons
    elements.forgotPasswordBtn?.addEventListener('click', () => switchTab('reset'));
    elements.backToLoginBtn?.addEventListener('click', () => switchTab('login'));
    
    // Password toggles
    elements.passwordToggles.forEach(toggle => {
        toggle.addEventListener('click', () => togglePasswordVisibility(toggle));
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            const activeForm = document.querySelector('.auth-form.active');
            if (activeForm) {
                const submitButton = activeForm.querySelector('button[type="submit"]');
                if (submitButton && !submitButton.disabled) {
                    submitButton.click();
                }
            }
        }
    });
}

function setupFormValidation() {
    // Real-time password confirmation validation
    if (elements.confirmPassword) {
        elements.confirmPassword.addEventListener('input', validatePasswordMatch);
        elements.registerPassword.addEventListener('input', validatePasswordMatch);
    }
    
    // Email format validation
    const emailInputs = [elements.loginEmail, elements.registerEmail, elements.resetEmail];
    emailInputs.forEach(input => {
        if (input) {
            input.addEventListener('blur', () => validateEmail(input));
        }
    });
    
    // Password strength validation
    if (elements.registerPassword) {
        elements.registerPassword.addEventListener('input', () => validatePasswordStrength(elements.registerPassword));
    }
}

function switchTab(tabName) {
    // Update tab buttons
    elements.tabButtons.forEach(button => {
        button.classList.toggle('active', button.dataset.tab === tabName);
    });
    
    // Show/hide forms
    const forms = document.querySelectorAll('.auth-form');
    forms.forEach(form => {
        form.classList.remove('active');
        form.style.display = 'none';
    });
    
    const activeForm = document.getElementById(`${tabName}-form`);
    if (activeForm) {
        activeForm.classList.add('active');
        activeForm.style.display = 'block';
        
        // Focus first input
        const firstInput = activeForm.querySelector('input[type="email"]');
        if (firstInput) {
            setTimeout(() => firstInput.focus(), 100);
        }
    }
    
    // Clear any error messages
    clearErrorMessages();
}

async function handleLogin(e) {
    e.preventDefault();
    
    const email = elements.loginEmail.value.trim();
    const password = elements.loginPassword.value;
    
    if (!validateLoginForm(email, password)) {
        return;
    }
    
    showLoading(true);
    setButtonLoading(elements.loginSubmit, true);
    
    try {
        const user = await AuthService.signIn(email, password);
        
        if (user.emailVerified) {
            showToast('Sign in successful! Redirecting...', 'success');
            
            // Get ID token and test API access
            const token = await user.getIdToken();
            await testApiAccess(token);
            
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 1500);
        } else {
            showToast('Please verify your email address before continuing', 'warning');
            showEmailVerificationPrompt(user);
        }
        
    } catch (error) {
        console.error('Login failed:', error);
        showAuthError(error);
    } finally {
        showLoading(false);
        setButtonLoading(elements.loginSubmit, false);
    }
}

async function handleRegister(e) {
    e.preventDefault();
    
    const email = elements.registerEmail.value.trim();
    const password = elements.registerPassword.value;
    const confirmPassword = elements.confirmPassword.value;
    
    if (!validateRegisterForm(email, password, confirmPassword)) {
        return;
    }
    
    showLoading(true);
    setButtonLoading(elements.registerSubmit, true);
    
    try {
        const user = await AuthService.signUp(email, password);
        
        showToast('Account created successfully! Please check your email for verification.', 'success');
        
        // Switch to login form
        setTimeout(() => {
            switchTab('login');
            elements.loginEmail.value = email;
        }, 2000);
        
    } catch (error) {
        console.error('Registration failed:', error);
        showAuthError(error);
    } finally {
        showLoading(false);
        setButtonLoading(elements.registerSubmit, false);
    }
}

async function handlePasswordReset(e) {
    e.preventDefault();
    
    const email = elements.resetEmail.value.trim();
    
    if (!validateEmail(elements.resetEmail)) {
        return;
    }
    
    showLoading(true);
    setButtonLoading(elements.resetSubmit, true);
    
    try {
        await AuthService.resetPassword(email);
        showToast('Password reset email sent! Check your inbox.', 'success');
        
        // Switch back to login after delay
        setTimeout(() => {
            switchTab('login');
        }, 3000);
        
    } catch (error) {
        console.error('Password reset failed:', error);
        showAuthError(error);
    } finally {
        showLoading(false);
        setButtonLoading(elements.resetSubmit, false);
    }
}

// Validation functions
function validateLoginForm(email, password) {
    let isValid = true;
    
    if (!email || !validateEmailFormat(email)) {
        showFieldError(elements.loginEmail, 'Please enter a valid email address');
        isValid = false;
    } else {
        clearFieldError(elements.loginEmail);
    }
    
    if (!password) {
        showFieldError(elements.loginPassword, 'Password is required');
        isValid = false;
    } else {
        clearFieldError(elements.loginPassword);
    }
    
    return isValid;
}

function validateRegisterForm(email, password, confirmPassword) {
    let isValid = true;
    
    // Email validation
    if (!email || !validateEmailFormat(email)) {
        showFieldError(elements.registerEmail, 'Please enter a valid university email address');
        isValid = false;
    } else {
        clearFieldError(elements.registerEmail);
    }
    
    // Password validation
    const passwordStrength = validatePasswordStrength(elements.registerPassword);
    if (!passwordStrength.isValid) {
        showFieldError(elements.registerPassword, passwordStrength.message);
        isValid = false;
    } else {
        clearFieldError(elements.registerPassword);
    }
    
    // Confirm password validation
    if (password !== confirmPassword) {
        showFieldError(elements.confirmPassword, 'Passwords do not match');
        isValid = false;
    } else if (confirmPassword) {
        clearFieldError(elements.confirmPassword);
    }
    
    return isValid;
}

function validateEmailFormat(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function validateEmail(input) {
    if (!input.value.trim()) return false;
    
    if (validateEmailFormat(input.value.trim())) {
        clearFieldError(input);
        return true;
    } else {
        showFieldError(input, 'Please enter a valid email address');
        return false;
    }
}

function validatePasswordStrength(input) {
    const password = input.value;
    const minLength = 8;
    
    if (password.length < minLength) {
        return {
            isValid: false,
            message: `Password must be at least ${minLength} characters long`
        };
    }
    
    if (!/(?=.*[a-z])/.test(password)) {
        return {
            isValid: false,
            message: 'Password must contain at least one lowercase letter'
        };
    }
    
    if (!/(?=.*[A-Z])/.test(password)) {
        return {
            isValid: false,
            message: 'Password must contain at least one uppercase letter'
        };
    }
    
    if (!/(?=.*\d)/.test(password)) {
        return {
            isValid: false,
            message: 'Password must contain at least one number'
        };
    }
    
    return { isValid: true, message: 'Password strength is good' };
}

function validatePasswordMatch() {
    const password = elements.registerPassword.value;
    const confirmPassword = elements.confirmPassword.value;
    
    if (confirmPassword && password !== confirmPassword) {
        showFieldError(elements.confirmPassword, 'Passwords do not match');
    } else if (confirmPassword) {
        clearFieldError(elements.confirmPassword);
    }
}

// UI Helper functions
function showFieldError(input, message) {
    input.classList.add('error');
    input.classList.remove('success');
    
    // Remove existing error message
    const existingError = input.parentNode.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    input.parentNode.appendChild(errorDiv);
}

function clearFieldError(input) {
    input.classList.remove('error');
    input.classList.add('success');
    
    const errorMessage = input.parentNode.querySelector('.error-message');
    if (errorMessage) {
        errorMessage.remove();
    }
}

function clearErrorMessages() {
    document.querySelectorAll('.error-message').forEach(error => error.remove());
    document.querySelectorAll('.form-group input').forEach(input => {
        input.classList.remove('error', 'success');
    });
}

function togglePasswordVisibility(toggle) {
    const targetId = toggle.dataset.target;
    const input = document.getElementById(targetId);
    const icon = toggle.querySelector('i');
    
    if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'fas fa-eye-slash';
    } else {
        input.type = 'password';
        icon.className = 'fas fa-eye';
    }
}

function showLoading(show) {
    elements.authLoading.style.display = show ? 'flex' : 'none';
}

function setButtonLoading(button, loading) {
    if (loading) {
        button.disabled = true;
        const originalText = button.innerHTML;
        button.dataset.originalText = originalText;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Please wait...';
    } else {
        button.disabled = false;
        if (button.dataset.originalText) {
            button.innerHTML = button.dataset.originalText;
        }
    }
}

function showAuthError(error) {
    let message = 'Authentication failed. Please try again.';
    
    switch (error.code) {
        case 'auth/user-not-found':
            message = 'No account found with this email address.';
            break;
        case 'auth/wrong-password':
            message = 'Incorrect password. Please try again.';
            break;
        case 'auth/email-already-in-use':
            message = 'An account with this email already exists.';
            break;
        case 'auth/weak-password':
            message = 'Password is too weak. Please choose a stronger password.';
            break;
        case 'auth/invalid-email':
            message = 'Please enter a valid email address.';
            break;
        case 'auth/too-many-requests':
            message = 'Too many failed attempts. Please try again later.';
            break;
        case 'auth/network-request-failed':
            message = 'Network error. Please check your connection.';
            break;
    }
    
    showToast(message, 'error');
}

function showEmailVerificationPrompt(user) {
    const verificationHtml = `
        <div class="auth-info" style="margin-top: 20px;">
            <p><i class="fas fa-envelope"></i> Please verify your email address to access the dashboard.</p>
            <button class="auth-button" id="resend-verification" style="margin-top: 15px;">
                <i class="fas fa-paper-plane"></i> Resend Verification Email
            </button>
        </div>
    `;
    
    // Add verification prompt to active form
    const activeForm = document.querySelector('.auth-form.active');
    if (activeForm && !activeForm.querySelector('#resend-verification')) {
        activeForm.insertAdjacentHTML('beforeend', verificationHtml);
        
        // Setup resend button
        document.getElementById('resend-verification').addEventListener('click', async () => {
            try {
                await AuthService.sendEmailVerification();
                showToast('Verification email sent!', 'success');
            } catch (error) {
                showToast('Failed to send verification email', 'error');
            }
        });
    }
}

async function testApiAccess(token) {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/auth/status`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ API access verified:', data);
        }
    } catch (error) {
        console.warn('⚠️ API access test failed:', error);
        // Don't block login if API is temporarily unavailable
    }
}

function handleUrlParameters() {
    const urlParams = new URLSearchParams(window.location.search);
    const tab = urlParams.get('tab');
    
    if (tab && ['login', 'register', 'reset'].includes(tab)) {
        switchTab(tab);
    }
}

// Toast notification function
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Configuration - this should match your backend URL
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000', // Backend URL
    REFRESH_INTERVAL: 10000,
};

console.log('✅ Authentication page loaded successfully!');
