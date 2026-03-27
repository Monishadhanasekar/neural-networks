# === MULTI-HOP NEEDLE ===
# The model must find TWO facts and connect them:
# Fact 1: "Dr. Elena Vasquez leads the AURORA-7 initiative."
# Fact 2: "Dr. Vasquez confirmed the launch window opens on March 15, 2024."
# Question: "What is the launch date for AURORA-7?"
# The model needs to connect Vasquez -> AURORA-7 -> March 15.

FACT_A = "Dr. Elena Vasquez was appointed as the lead researcher for the classified AURORA-7 initiative after her successful tenure directing the Meridian program."
FACT_B = "In a private briefing last Tuesday, Dr. Vasquez confirmed that the launch window for her current project opens on March 15, 2024, pending final safety reviews."

# Filler paragraphs about similar topics (projects, dates, people)
FILLER_POOL = [
    "The quarterly infrastructure review highlighted upgrades to the east-coast data centers. Migration timelines were set for late Q2. The operations team reported a 99.97 percent uptime for the previous period and proposed additional redundancy measures for the backup systems.",
    "Dr. James Morton presented findings from the computational biology lab. The new protein folding algorithm showed a 34 percent improvement in prediction accuracy. Funding proposals were submitted to three federal agencies for continued research.",
    "Project TITAN-6 entered Phase 3 clinical evaluation under the supervision of Dr. Sarah Lin. Preliminary results are expected by August 2024. The regulatory affairs team has begun preparing submission documents for the oversight board.",
    "The advanced materials team tested a new ceramic composite for thermal shielding. Results exceeded expectations at temperatures above 2000 degrees Celsius. A patent application was filed and manufacturing partners have been contacted.",
    "Chief Financial Officer Mark Reynolds presented the revised budget allocations for fiscal year 2025. Research and development spending will increase by 18 percent. Capital expenditure for new laboratory facilities was approved at 47 million dollars.",
    "The cybersecurity division completed its annual red team exercise across all classified networks. Two medium-severity vulnerabilities were discovered and patched within 72 hours. New endpoint detection protocols were deployed organization-wide.",
    "Professor Lisa Chang published her team's findings on quantum entanglement stability in the latest issue of Physical Review Letters. The results suggest a viable path toward error-corrected quantum computing within the next decade.",
    "Project ORION-11 received continued funding approval from the defense advisory panel. The next milestone review is scheduled for May 2024. Program manager David Kowalski noted that the project remains on schedule and within budget parameters.",
    "The talent acquisition team reported hiring 43 new researchers across six departments. Retention rates improved to 94 percent following the introduction of flexible work arrangements. The mentorship program expanded to include 120 pairs.",
    "Environmental monitoring stations detected a slight increase in seismic activity near the northern test facility. Geological surveys confirmed no structural risk. Additional monitoring equipment has been installed as a precautionary measure.",
    "The satellite communications upgrade was completed two weeks ahead of schedule. Bandwidth capacity doubled to support the growing number of field operations. Dr. Robert Kim oversaw the final integration testing phase and signed off on deployment.",
    "Project NEBULA-4 concluded its preliminary design review. The propulsion subsystem exceeded thrust-to-weight requirements by 12 percent. Systems engineering lead Patricia Gomez recommended advancing to the detailed design phase in Q3.",
    "The medical research wing reported promising results from the Phase 2 trial of compound MRX-7821. Efficacy rates reached 67 percent in the target population. An expanded trial involving 2000 additional participants was approved for the next quarter.",
    "Facilities management completed the renovation of Building 7, adding 15000 square feet of laboratory space. The new cleanroom meets ISO Class 5 standards. Occupancy is planned for early February 2024.",
    "Dr. Amanda Foster's team demonstrated a novel machine learning approach for anomaly detection in network traffic. The system reduced false positive rates by 40 percent compared to existing methods used across the organization.",
    "The supply chain optimization initiative yielded a 22 percent reduction in procurement lead times. Strategic partnerships with three new vendors were established. Inventory carrying costs decreased by 1.3 million dollars annually.",
    "The HELIOS-8 solar energy research program completed its first full year of field testing. Energy conversion efficiency averaged 31.4 percent across all test sites. Results will be presented at the International Energy Conference in Geneva.",
    "Human resources launched the updated professional development framework. Over 300 employees enrolled in the advanced leadership track. The tuition reimbursement budget was increased by 25 percent to accommodate growing demand.",
    "The autonomous systems laboratory tested its latest navigation algorithm in simulated urban environments. The system achieved a 98.7 percent obstacle avoidance rate. Real-world field trials are planned for the second half of 2024.",
    "Project AEGIS-3 was formally closed after achieving all deliverables ahead of schedule. The final report documented 14 technical innovations and 6 patent filings. Lessons learned sessions were conducted with all participating teams."
]

print(f"Fact A: {FACT_A}")
print(f"Fact B: {FACT_B}")
print(f"Filler paragraphs available: {len(FILLER_POOL)}")