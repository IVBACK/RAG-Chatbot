# Deployment Guide (Nginx + SSL)

This guide helps you configure your Flask-based AI chat API in a production environment using **Nginx** and **Let's Encrypt SSL**.

---

## ✅ Prerequisites

```bash
sudo apt install nginx certbot python3-certbot-nginx -y
sudo certbot --nginx -d example.com
```

> If successful, certificates will be stored at:  
> `/etc/letsencrypt/live/example.com/`

---

## ✅ Global Rate Limiting (Optional)

Add this block to your `/etc/nginx/nginx.conf` inside the `http {}` block:

```nginx
limit_req_zone $binary_remote_addr zone=chat_limit:10m rate=5r/s;
```

---

## ✅ Enable your Nginx Site

```bash
sudo cp deployment/site.example.com.conf /etc/nginx/sites-available/chat
sudo ln -s /etc/nginx/sites-available/chat /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## ✅ Favicon

```bash
sudo mkdir -p /var/www/static
sudo cp deployment/static/favicon.png /var/www/static/
sudo chmod 644 /var/www/static/favicon.png
```

> Customize the `favicon.png` in the `deployment/static/` folder as needed.
