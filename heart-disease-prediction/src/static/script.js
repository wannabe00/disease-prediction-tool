// script.js - Member 4: Levan Bokuchava (UPDATED WITH DEEPSEEK)

document.addEventListener("DOMContentLoaded", function () {
	// Initialize the application
	initializeApp();
});

function initializeApp() {
	// Set up event listeners
	setupEventListeners();

	// Initialize UI components
	updateAgeDisplay();

	// Load model information
	loadModelInfo();
}

function setupEventListeners() {
	// Age slider
	const ageSlider = document.getElementById("age");
	ageSlider.addEventListener("input", updateAgeDisplay);

	// Form submission
	const form = document.getElementById("predictionForm");
	form.addEventListener("submit", handleFormSubmission);

	// Real-time validation
	setupFormValidation();
}

function updateAgeDisplay() {
	const ageSlider = document.getElementById("age");
	const ageValue = document.getElementById("ageValue");
	ageValue.textContent = ageSlider.value;
}

function setupFormValidation() {
	const inputs = document.querySelectorAll('input[type="number"]');
	inputs.forEach((input) => {
		input.addEventListener("input", function () {
			validateInput(this);
		});
	});
}

function validateInput(input) {
	const value = parseFloat(input.value);
	const min = parseFloat(input.min);
	const max = parseFloat(input.max);

	if (value < min || value > max) {
		input.classList.add("is-invalid");
	} else {
		input.classList.remove("is-invalid");
	}
}

function handleFormSubmission(event) {
	event.preventDefault();

	// Show loading state
	showLoading();

	// Collect form data
	const formData = collectFormData();

	// Validate form data
	if (!validateFormData(formData)) {
		hideLoading();
		showError("Please fill in all required fields correctly.");
		return;
	}

	// Make prediction request
	makePrediction(formData);
}

function collectFormData() {
	const form = document.getElementById("predictionForm");
	const formData = new FormData(form);

	const data = {};
	for (let [key, value] of formData.entries()) {
		data[key] = value;
	}

	return data;
}

function validateFormData(data) {
	// Basic validation
	const requiredFields = [
		"age",
		"sex",
		"chest_pain",
		"blood_pressure",
		"cholesterol",
		"max_heart_rate",
	];

	for (let field of requiredFields) {
		if (!data[field]) {
			return false;
		}
	}

	// Range validation
	const validations = {
		age: [20, 80],
		blood_pressure: [80, 200],
		cholesterol: [100, 400],
		max_heart_rate: [60, 220],
		st_depression: [0, 6],
	};

	for (let [field, [min, max]] of Object.entries(validations)) {
		if (data[field]) {
			const value = parseFloat(data[field]);
			if (value < min || value > max) {
				return false;
			}
		}
	}

	return true;
}

function makePrediction(data) {
	fetch("/predict", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(data),
	})
		.then((response) => response.json())
		.then((result) => {
			hideLoading();
			if (result.error) {
				showError(result.error);
			} else {
				showResults(result, data); // Pass form data for AI recommendations
			}
		})
		.catch((error) => {
			hideLoading();
			showError("Network error. Please try again.");
			console.error("Error:", error);
		});
}

function showLoading() {
	document.getElementById("loadingSpinner").classList.remove("d-none");
	document.getElementById("resultsSection").classList.add("d-none");
	document.getElementById("errorSection").classList.add("d-none");
}

function hideLoading() {
	document.getElementById("loadingSpinner").classList.add("d-none");
}

function showResults(result, formData) {
	// Show results section
	const resultsSection = document.getElementById("resultsSection");
	resultsSection.classList.remove("d-none");
	resultsSection.classList.add("fade-in-up");

	// Update risk level
	updateRiskLevel(result.risk_level, result.probability);

	// Update probability gauge
	updateProbabilityGauge(result.probability);

	// Show model comparisons
	showModelComparisons(result.all_models, result.best_model);

	// Show basic recommendations
	showRecommendations(result.risk_level, result.probability);

	// ===== NEW: Get AI-powered recommendations =====
	getAIRecommendations(result, formData);
}

// ===== NEW DEEPSEEK AI RECOMMENDATION FUNCTIONS =====

async function getAIRecommendations(predictionResult, patientData) {
	try {
		// Show AI loading indicator
		showAILoading();

		const response = await fetch("/recommendations", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				prediction_result: predictionResult,
				patient_data: patientData,
			}),
		});

		if (response.ok) {
			const recommendations = await response.json();
			displayAIRecommendations(recommendations);
		} else {
			console.log("AI recommendations not available, using fallback");
			showAIError("AI recommendations temporarily unavailable");
		}
	} catch (error) {
		console.log("AI recommendations failed:", error);
		showAIError("Unable to generate AI recommendations");
	}
}

function showAILoading() {
	const aiSection = document.getElementById("aiRecommendations");
	if (aiSection) {
		aiSection.innerHTML = `
            <div class="alert alert-info">
                <div class="d-flex align-items-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <span>Generating AI-powered health recommendations...</span>
                </div>
            </div>
        `;
	}
}

function displayAIRecommendations(recommendations) {
	const aiSection = document.getElementById("aiRecommendations");
	if (aiSection) {
		// Format recommendations with better styling
		const formattedRecommendations = recommendations.recommendations
			.split("\n")
			.filter((line) => line.trim())
			.map((line) => {
				// If line starts with number, make it a list item
				if (line.match(/^\d+\./)) {
					return `<li class="mb-2">${line.replace(/^\d+\.\s*/, "")}</li>`;
				}
				return `<p class="mb-2">${line}</p>`;
			})
			.join("");

		const usageInfo = recommendations.usage
			? `<small class="text-muted">API Usage: ${
					recommendations.usage.tokens
			  } tokens (~$${recommendations.usage.cost.toFixed(6)})</small>`
			: "";

		aiSection.innerHTML = `
            <div class="alert alert-success border-0 shadow-sm">
                <div class="d-flex align-items-center mb-3">
                    <i class="fas fa-robot text-primary me-2" style="font-size: 1.2em;"></i>
                    <h6 class="mb-0 fw-bold">AI-Powered Health Recommendations</h6>
                    <span class="badge bg-primary ms-auto">DeepSeek AI</span>
                </div>
                
                <div class="recommendations-content">
                    <ol class="mb-3">
                        ${formattedRecommendations}
                    </ol>
                </div>
                
                <div class="border-top pt-3">
                    <small class="text-muted d-block">
                        <strong>Source:</strong> ${recommendations.source}<br>
                        <strong>Disclaimer:</strong> ${recommendations.disclaimer}
                    </small>
                    ${usageInfo}
                </div>
            </div>
        `;

		// Add animation
		aiSection.classList.add("fade-in-up");
	}
}

function showAIError(message) {
	const aiSection = document.getElementById("aiRecommendations");
	if (aiSection) {
		aiSection.innerHTML = `
            <div class="alert alert-warning border-0">
                <div class="d-flex align-items-center">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <span>${message}</span>
                </div>
                <small class="text-muted mt-2 d-block">
                    Basic evidence-based recommendations are shown above.
                </small>
            </div>
        `;
	}
}

// ===== EXISTING FUNCTIONS (UPDATED) =====

function updateRiskLevel(riskLevel, probability) {
	const riskAlert = document.getElementById("riskAlert");
	const riskLevelElement = document.getElementById("riskLevel");
	const riskDescription = document.getElementById("riskDescription");

	// Remove existing classes
	riskAlert.className = "alert";

	// Set content and styling based on risk level
	riskLevelElement.textContent = riskLevel;

	const probabilityPercent = (probability * 100).toFixed(1);

	switch (riskLevel) {
		case "Low Risk":
			riskAlert.classList.add("risk-low");
			riskDescription.innerHTML = `
                <strong>Good news!</strong> Your cardiovascular risk is low (${probabilityPercent}% probability).
                <br>Continue maintaining your healthy lifestyle.
            `;
			break;
		case "Moderate Risk":
			riskAlert.classList.add("risk-moderate");
			riskDescription.innerHTML = `
                <strong>Moderate risk detected.</strong> Your cardiovascular risk is ${probabilityPercent}%.
                <br>Consider lifestyle improvements and consult your doctor.
            `;
			break;
		case "High Risk":
			riskAlert.classList.add("risk-high");
			riskDescription.innerHTML = `
                <strong>High risk detected.</strong> Your cardiovascular risk is ${probabilityPercent}%.
                <br>Please consult a healthcare professional immediately.
            `;
			break;
	}
}

function updateProbabilityGauge(probability) {
	const canvas = document.getElementById("probabilityGauge");
	const ctx = canvas.getContext("2d");
	const probabilityValue = document.getElementById("probabilityValue");

	// Clear canvas
	ctx.clearRect(0, 0, canvas.width, canvas.height);

	// Set up gauge parameters
	const centerX = canvas.width / 2;
	const centerY = canvas.height - 20;
	const radius = 100;
	const startAngle = Math.PI;
	const endAngle = 2 * Math.PI;

	// Draw gauge background
	ctx.beginPath();
	ctx.arc(centerX, centerY, radius, startAngle, endAngle);
	ctx.lineWidth = 20;
	ctx.strokeStyle = "#e9ecef";
	ctx.stroke();

	// Draw probability arc
	const probabilityAngle = startAngle + probability * Math.PI;
	ctx.beginPath();
	ctx.arc(centerX, centerY, radius, startAngle, probabilityAngle);
	ctx.lineWidth = 20;

	// Set color based on probability
	if (probability < 0.3) {
		ctx.strokeStyle = "#38ef7d";
	} else if (probability < 0.6) {
		ctx.strokeStyle = "#f5576c";
	} else {
		ctx.strokeStyle = "#fc466b";
	}
	ctx.stroke();

	// Draw center text
	ctx.fillStyle = "#333";
	ctx.font = "bold 24px Arial";
	ctx.textAlign = "center";
	const percentText = (probability * 100).toFixed(1) + "%";
	ctx.fillText(percentText, centerX, centerY - 10);

	// Update probability value
	probabilityValue.textContent = percentText;
}

function showModelComparisons(allModels, bestModel) {
	const modelResults = document.getElementById("modelResults");
	modelResults.innerHTML = "";

	for (let [modelName, modelData] of Object.entries(allModels)) {
		const isBest = modelName === bestModel;
		const probability = (modelData.probability * 100).toFixed(1);
		const prediction = modelData.prediction === 1 ? "Disease" : "No Disease";

		const modelDiv = document.createElement("div");
		modelDiv.className = "model-result";
		if (isBest) {
			modelDiv.style.borderLeftColor = "#28a745";
			modelDiv.style.fontWeight = "bold";
		}

		modelDiv.innerHTML = `
            <div class="d-flex justify-content-between">
                <span>${modelName} ${isBest ? "(Best)" : ""}</span>
                <span>${prediction} (${probability}%)</span>
            </div>
            <div class="progress mt-1" style="height: 5px;">
                <div class="progress-bar" style="width: ${probability}%"></div>
            </div>
        `;

		modelResults.appendChild(modelDiv);
	}
}

function showRecommendations(riskLevel, probability) {
	const recommendations = document.getElementById("recommendations");
	recommendations.innerHTML = "";

	let recommendationList = [];

	if (riskLevel === "Low Risk") {
		recommendationList = [
			"Maintain regular physical activity (150 minutes/week)",
			"Continue healthy diet with fruits and vegetables",
			"Regular health check-ups annually",
			"Avoid smoking and limit alcohol consumption",
		];
	} else if (riskLevel === "Moderate Risk") {
		recommendationList = [
			"Increase physical activity and exercise regularly",
			"Adopt a heart-healthy diet (Mediterranean style)",
			"Monitor blood pressure and cholesterol regularly",
			"Consider stress management techniques",
			"Consult your doctor for detailed evaluation",
		];
	} else {
		recommendationList = [
			"Seek immediate medical consultation",
			"Consider cardiac evaluation and stress testing",
			"Implement strict dietary modifications",
			"Start supervised exercise program",
			"Monitor vital signs regularly",
			"Follow medication regimen if prescribed",
		];
	}

	recommendationList.forEach((recommendation) => {
		const li = document.createElement("li");
		li.textContent = recommendation;
		recommendations.appendChild(li);
	});

	// ===== NEW: Add AI recommendations section =====
	if (!document.getElementById("aiRecommendations")) {
		const aiDiv = document.createElement("div");
		aiDiv.id = "aiRecommendations";
		aiDiv.className = "mt-4";

		// Insert after basic recommendations
		const recommendationsContainer = recommendations.closest(".alert");
		recommendationsContainer.parentNode.insertBefore(
			aiDiv,
			recommendationsContainer.nextSibling
		);
	}
}

function showError(message) {
	const errorSection = document.getElementById("errorSection");
	const errorMessage = document.getElementById("errorMessage");

	errorMessage.textContent = message;
	errorSection.classList.remove("d-none");
	errorSection.classList.add("fade-in-up");
}

function loadModelInfo() {
	fetch("/model_info")
		.then((response) => response.json())
		.then((data) => {
			console.log("Model info loaded:", data);

			// Show DeepSeek status in UI if available
			if (data.deepseek_enabled) {
				console.log("✅ DeepSeek AI recommendations enabled");
			} else {
				console.log("⚠️ DeepSeek AI not configured");
			}
		})
		.catch((error) => {
			console.log("Could not load model info:", error);
		});
}

// ===== NEW: USAGE STATISTICS FUNCTION =====
async function loadUsageStats() {
	try {
		const response = await fetch("/usage");
		const data = await response.json();

		console.log(
			`Daily usage: ${
				data.daily_requests
			} requests, ${data.daily_cost.toFixed(6)} cost`
		);

		// You can display this in the UI if wanted
		return data;
	} catch (error) {
		console.log("Could not load usage stats:", error);
		return null;
	}
}

// Utility functions
function formatNumber(num, decimals = 2) {
	return Number(num).toFixed(decimals);
}

function debounce(func, wait) {
	let timeout;
	return function executedFunction(...args) {
		const later = () => {
			clearTimeout(timeout);
			func(...args);
		};
		clearTimeout(timeout);
		timeout = setTimeout(later, wait);
	};
}

// Add smooth animations
function addAnimation(element, animationClass) {
	element.classList.add(animationClass);
	element.addEventListener("animationend", () => {
		element.classList.remove(animationClass);
	});
}

// Form reset function
function resetForm() {
	document.getElementById("predictionForm").reset();
	updateAgeDisplay();
	hideAllSections();
}

function hideAllSections() {
	document.getElementById("loadingSpinner").classList.add("d-none");
	document.getElementById("resultsSection").classList.add("d-none");
	document.getElementById("errorSection").classList.add("d-none");

	// Clear AI recommendations
	const aiSection = document.getElementById("aiRecommendations");
	if (aiSection) {
		aiSection.innerHTML = "";
	}
}

// Add tooltips for better UX
function initializeTooltips() {
	const tooltipTriggerList = [].slice.call(
		document.querySelectorAll('[data-bs-toggle="tooltip"]')
	);
	tooltipTriggerList.map(function (tooltipTriggerEl) {
		return new bootstrap.Tooltip(tooltipTriggerEl);
	});
}

// ===== NEW: DEMO FUNCTION FOR TESTING =====
function runDemoTest() {
	const demoData = {
		age: "65",
		sex: "male",
		chest_pain: "1",
		blood_pressure: "160",
		cholesterol: "280",
		fasting_sugar: "no",
		rest_ecg: "0",
		max_heart_rate: "120",
		exercise_angina: "yes",
		st_depression: "2.0",
		slope: "2",
		vessels: "2",
		thalassemia: "2",
	};

	console.log("Running demo test with sample data...");
	makePrediction(demoData);
}

// Add demo button functionality (you can add this to your HTML if you want)
if (window.location.search.includes("demo=true")) {
	setTimeout(() => {
		runDemoTest();
	}, 1000);
}
