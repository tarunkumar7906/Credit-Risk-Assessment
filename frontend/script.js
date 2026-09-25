// ===============================
// Get HTML elements
// ===============================

const form = document.getElementById('riskForm');
const submitBtn = document.getElementById('submitBtn');
const formError = document.getElementById('formError');

const resultEmpty = document.getElementById('resultEmpty');
const resultLoading = document.getElementById('resultLoading');
const resultContent = document.getElementById('resultContent');
const resultErrorBox = document.getElementById('resultErrorBox');
const resultErrorDetail = document.getElementById('resultErrorDetail');

const gaugeFill = document.getElementById('gaugeFill');
const probValue = document.getElementById('probValue');
const verdict = document.getElementById('verdict');
const verdictDot = document.getElementById('verdictDot');
const verdictText = document.getElementById('verdictText');
const thresholdValue = document.getElementById('thresholdValue');
const predictionValue = document.getElementById('predictionValue');


// ===============================
// FastAPI endpoint
// ===============================

const DEFAULT_API = 'https://credit-risk-assessment-o5qn.onrender.com/predict';


// ===============================
// Show different UI states
// ===============================

function showState(state) {
    resultEmpty.hidden = state !== 'empty';
    resultLoading.hidden = state !== 'loading';
    resultContent.hidden = state !== 'result';
    resultErrorBox.hidden = state !== 'error';
}


// ===============================
// Animate probability number
// ===============================

function animateNumber(el, to, duration = 900) {
    const start = performance.now();
    const from = 0;

    function tick(now) {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);

        const val = from + (to - from) * eased;

        el.textContent = val.toFixed(1);

        if (t < 1) {
            requestAnimationFrame(tick);
        } else {
            el.textContent = to.toFixed(1);
        }
    }

    requestAnimationFrame(tick);
}


// ===============================
// Gauge
// ===============================

function setGauge(fraction) {

    const GAUGE_LENGTH = 251.2;

    // Keep value between 0 and 1
    fraction = Math.min(1, Math.max(0, fraction));

    const offset = GAUGE_LENGTH * (1 - fraction);

    gaugeFill.style.transition = 'none';
    gaugeFill.style.strokeDashoffset = GAUGE_LENGTH;

    // Force browser reflow
    void gaugeFill.getBoundingClientRect();

    gaugeFill.style.transition = '';

    requestAnimationFrame(() => {
        gaugeFill.style.strokeDashoffset = String(offset);
    });
}


// ===============================
// Form submission
// ===============================

form.addEventListener('submit', async (e) => {

    // Don't reload the page
    e.preventDefault();

    // Clear previous error
    formError.textContent = '';

    // Show loading
    submitBtn.disabled = true;
    submitBtn.classList.add('is-loading');

    showState('loading');


    // ===============================
    // Get form data
    // ===============================

    const fd = new FormData(form);


    // ===============================
    // Create payload
    // ===============================

    const payload = {

        person_age: Number(
            fd.get('person_age')
        ),

        person_income: Number(
            fd.get('person_income')
        ),

        person_home_ownership:
            fd.get('person_home_ownership'),

        person_emp_length: Number(
            fd.get('person_emp_length')
        ),

        loan_intent:
            fd.get('loan_intent'),

        loan_grade:
            fd.get('loan_grade'),

        loan_amnt: Number(
            fd.get('loan_amnt')
        ),

        loan_int_rate: Number(
            fd.get('loan_int_rate')
        ),

        loan_percent_income: Number(
            fd.get('loan_percent_income')
        ),

        cb_person_default_on_file:
            fd.get('cb_person_default_on_file'),

        cb_person_cred_hist_length: Number(
            fd.get('cb_person_cred_hist_length')
        )
    };


    // ===============================
    // Validate input
    // ===============================

    for (const [key, val] of Object.entries(payload)) {

        if (
            val === '' ||
            val === null ||
            (
                typeof val === 'number' &&
                Number.isNaN(val)
            )
        ) {

            formError.textContent =
                'Please fill in every field before submitting.';

            submitBtn.disabled = false;
            submitBtn.classList.remove('is-loading');

            showState('empty');

            return;
        }
    }


    // ===============================
    // Send data to FastAPI
    // ===============================

    try {

        console.log('Sending data to FastAPI:');
        console.log(payload);

        const response = await fetch(DEFAULT_API, {

            method: 'POST',

            headers: {
                'Content-Type': 'application/json'
            },

            body: JSON.stringify(payload)
        });


        // ===============================
        // Check server response
        // ===============================

        if (!response.ok) {

            const text = await response.text().catch(() => '');

            throw new Error(
                `Server responded with ${response.status}` +
                `${text ? `: ${text}` : ''}`
            );
        }


        // ===============================
        // Convert response to JSON
        // ===============================

        const data = await response.json();

        console.log('Response from FastAPI:');
        console.log(data);


        // ===============================
        // Get prediction values
        // ===============================

        const probability = Number(
            data.default_probability ?? 0
        );

        const prediction = Number(
            data.default_prediction ?? 0
        );

        const threshold = Number(
            data.threshold ?? 0
        );

        const resultLabel =
            data.Result ||
            (
                prediction === 1
                    ? 'High Risk'
                    : 'Low Risk'
            );


        // Probability → percentage

        const pct = probability * 100;


        // ===============================
        // Display result
        // ===============================

        showState('result');


        // Gauge

        setGauge(
            Math.min(
                1,
                Math.max(0, probability)
            )
        );


        // Probability number

        animateNumber(
            probValue,
            pct
        );


        // High/Low risk

        const isHigh = prediction === 1;

        verdict.classList.toggle(
            'is-high',
            isHigh
        );


        // Reset dot styling

        verdictDot.style.background = '';


        // Result text

        verdictText.textContent =
            resultLabel;


        // Threshold

        thresholdValue.textContent =
            (threshold * 100).toFixed(1) + '%';


        // Prediction

        predictionValue.textContent =
            isHigh
                ? 'Default (1)'
                : 'No default (0)';

    }


    // ===============================
    // Error handling
    // ===============================

    catch (error) {

        console.error(
            'Prediction error:',
            error
        );

        showState('error');

        resultErrorDetail.textContent =
            error && error.message
                ? error.message
                : 'Unable to connect to the prediction API.';
    }


    // ===============================
    // Enable button again
    // ===============================

    finally {

        submitBtn.disabled = false;

        submitBtn.classList.remove(
            'is-loading'
        );
    }

});