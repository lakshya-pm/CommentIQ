/**
 * ===========================================================
 * CommentIQ - NLP Sentiment Analysis
 * Client-Side JavaScript
 * ===========================================================
 *
 * Features:
 * ✔ Form submission loading state
 * ✔ Smooth scroll navigation
 * ✔ Intersection Observer animations
 * ✔ Navbar scroll effect
 * ✔ Number counter animation
 */

document.addEventListener("DOMContentLoaded", function () {

    // -------------------------------------------------
    // Form Submission — Loading State
    // -------------------------------------------------

    const form = document.getElementById("analyzeForm");
    const btn = document.getElementById("analyzeBtn");

    if (form && btn) {
        form.addEventListener("submit", function () {
            btn.classList.add("loading");
        });
    }


    // -------------------------------------------------
    // Smooth Scroll for Anchor Links
    // -------------------------------------------------

    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();

            const target = document.querySelector(this.getAttribute("href"));

            if (target) {
                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    });


    // -------------------------------------------------
    // Navbar Scroll Shadow Effect
    // -------------------------------------------------

    const navbar = document.querySelector(".navbar");

    if (navbar) {
        window.addEventListener("scroll", function () {
            if (window.scrollY > 10) {
                navbar.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.06)";
            } else {
                navbar.style.boxShadow = "none";
            }
        });
    }


    // -------------------------------------------------
    // Intersection Observer — Animate on Scroll
    // -------------------------------------------------

    const observerOptions = {
        root: null,
        rootMargin: "0px",
        threshold: 0.1
    };

    const observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.style.opacity = "1";
                entry.target.style.transform = "translateY(0)";
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe workflow steps and feature cards
    document.querySelectorAll(
        ".workflow-step, .feature-card, .chart-card, .comment-card, .pipeline-card"
    ).forEach(function (el) {
        observer.observe(el);
    });


    // -------------------------------------------------
    // Stat Card Number Animation
    // -------------------------------------------------

    const statValues = document.querySelectorAll(".stat-value");

    statValues.forEach(function (el) {
        const finalValue = parseInt(el.textContent);

        if (isNaN(finalValue)) return;

        el.textContent = "0";

        const duration = 1200;
        const startTime = performance.now();

        function animateCount(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);

            el.textContent = Math.round(eased * finalValue);

            if (progress < 1) {
                requestAnimationFrame(animateCount);
            }
        }

        // Start counting when visible
        const countObserver = new IntersectionObserver(function (entries) {
            if (entries[0].isIntersecting) {
                requestAnimationFrame(animateCount);
                countObserver.unobserve(el);
            }
        }, { threshold: 0.5 });

        countObserver.observe(el);
    });


    // -------------------------------------------------
    // URL Input Validation Visual
    // -------------------------------------------------

    const urlInput = document.getElementById("urlInput");

    if (urlInput) {
        urlInput.addEventListener("input", function () {
            const value = this.value.trim();
            const isYouTube = value.includes("youtube.com") || value.includes("youtu.be");
            const isReddit = value.includes("reddit.com");

            if (isYouTube) {
                this.style.borderColor = "#dc2626";
            } else if (isReddit) {
                this.style.borderColor = "#ea580c";
            } else if (value.length > 0) {
                this.style.borderColor = "#e2e8f0";
            } else {
                this.style.borderColor = "#e2e8f0";
            }
        });
    }

});