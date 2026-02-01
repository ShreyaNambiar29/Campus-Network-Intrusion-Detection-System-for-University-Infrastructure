# Deployment Guide - Campus Network IDS

## Quick Start

### 1. MongoDB Atlas Setup
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free account and cluster
3. Create a database user
4. Get your connection string
5. Whitelist IP addresses (0.0.0.0/0 for development)

### 2. Backend Deployment (Choose One)

#### Option A: Deploy to Render (Recommended)
1. Fork this repository
2. Go to [Render.com](https://render.com) and connect your GitHub
3. Create a new Web Service
4. Select your forked repository
5. Use these settings:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable:
   - `MONGODB_URI`: Your MongoDB Atlas connection string
7. Deploy and get your backend URL

#### Option B: Deploy to Railway
1. Fork this repository
2. Go to [Railway.app](https://railway.app) and connect GitHub
3. Deploy the `backend` folder
4. Add environment variable: `MONGODB_URI`
5. Get your deployment URL

### 3. Frontend Deployment

#### Option A: Deploy to Vercel (Recommended)
1. Go to [Vercel.com](https://vercel.com) and connect GitHub
2. Import your forked repository
3. Set **Root Directory** to `frontend`
4. Before deploying, update `frontend/js/app.js`:
   ```javascript
   const CONFIG = {
       API_BASE_URL: 'https://your-backend-url.onrender.com', // Your actual backend URL
       REFRESH_INTERVAL: 10000,
   };
   ```
5. Deploy

#### Option B: Deploy to Netlify
1. Go to [Netlify.com](https://netlify.com)
2. Connect your GitHub repository
3. Set **Publish directory** to `frontend`
4. Update the API URL in `frontend/js/app.js`
5. Deploy

## Environment Variables

### Backend (.env)
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/campus_ids?retryWrites=true&w=majority
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production
```

### Frontend (js/app.js)
```javascript
const CONFIG = {
    API_BASE_URL: 'https://your-backend-url.onrender.com',
    REFRESH_INTERVAL: 10000,
};
```

## Testing Your Deployment

1. Visit your frontend URL
2. Check that the dashboard loads
3. Click "Simulate Attack" to test backend connectivity
4. Verify alerts appear in the table
5. Test the "Resolve" functionality
6. Check auto-refresh is working

## Troubleshooting

### Backend Issues
- Check logs in your deployment platform
- Verify MONGODB_URI is correct
- Test the `/health` endpoint
- Check `/docs` for API documentation

### Frontend Issues
- Check browser console for JavaScript errors
- Verify API_BASE_URL points to your backend
- Check CORS settings if needed
- Test backend endpoints directly

### CORS Issues
If you get CORS errors, the backend is configured to allow common origins. Make sure your frontend domain is accessible.

## Production Checklist

- [ ] MongoDB Atlas cluster configured
- [ ] Backend deployed and accessible
- [ ] Frontend deployed and accessible  
- [ ] Environment variables set
- [ ] API_BASE_URL updated in frontend
- [ ] Test all functionality
- [ ] Monitor logs for errors

## Security Notes

For production deployment:
1. Use strong MongoDB passwords
2. Restrict MongoDB IP whitelist
3. Use HTTPS for both frontend and backend
4. Consider adding authentication
5. Monitor logs and set up alerts

## Support

If you encounter issues:
1. Check the logs in your deployment platform
2. Test endpoints with the `/docs` interface
3. Verify environment variables are set correctly
4. Check the GitHub repository for updates
