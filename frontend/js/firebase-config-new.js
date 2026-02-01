// Firebase Configuration
// Replace with your actual Firebase project configuration

const firebaseConfig = {
    apiKey: "AIzaSyDgCr_dFc10CieVpkqWNHEPiHHjC6iDbBM",
    authDomain: "campus-network-ids.firebaseapp.com",
    projectId: "campus-network-ids",
    storageBucket: "campus-network-ids.firebasestorage.app",
    messagingSenderId: "267690223141",
    appId: "1:267690223141:web:2c5d77f5f1e4ae507f01cc",
    measurementId: "G-EH84LM6MQL"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Export auth instance
const auth = firebase.auth();

// Auth state persistence
auth.setPersistence(firebase.auth.Auth.Persistence.SESSION);

// Authentication helper functions
const AuthService = {
    // Get current user's ID token
    async getIdToken() {
        const user = auth.currentUser;
        if (user) {
            return await user.getIdToken();
        }
        return null;
    },

    // Get current user
    getCurrentUser() {
        return auth.currentUser;
    },

    // Sign in with email and password
    async signIn(email, password) {
        try {
            const result = await auth.signInWithEmailAndPassword(email, password);
            return result.user;
        } catch (error) {
            throw error;
        }
    },

    // Sign up with email and password
    async signUp(email, password) {
        try {
            const result = await auth.createUserWithEmailAndPassword(email, password);
            
            // Send email verification
            await result.user.sendEmailVerification();
            
            return result.user;
        } catch (error) {
            throw error;
        }
    },

    // Sign out
    async signOut() {
        try {
            await auth.signOut();
        } catch (error) {
            throw error;
        }
    },

    // Send password reset email
    async resetPassword(email) {
        try {
            await auth.sendPasswordResetEmail(email);
        } catch (error) {
            throw error;
        }
    },

    // Send email verification
    async sendEmailVerification() {
        const user = auth.currentUser;
        if (user && !user.emailVerified) {
            await user.sendEmailVerification();
        }
    },

    // Check if user is signed in
    isSignedIn() {
        return !!auth.currentUser;
    },

    // Listen to auth state changes
    onAuthStateChanged(callback) {
        return auth.onAuthStateChanged(callback);
    },

    // Get auth headers for API requests
    async getAuthHeaders() {
        const token = await this.getIdToken();
        if (token) {
            return {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            };
        }
        return {
            'Content-Type': 'application/json'
        };
    },

    // Make authenticated API request
    async apiRequest(url, options = {}) {
        const headers = await this.getAuthHeaders();
        
        const requestOptions = {
            ...options,
            headers: {
                ...headers,
                ...options.headers
            }
        };

        return fetch(url, requestOptions);
    }
};

// Export for use in other files
window.AuthService = AuthService;
