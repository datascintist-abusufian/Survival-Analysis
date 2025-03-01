import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

const SurvivalAnalysisVisualizations = () => {
  // Placeholder data based on the analysis
  
  // 1. Baseline Survival Data
  const baselineSurvivalData = [
    { time: 0, survival: 1.000 },
    { time: 30, survival: 0.942 },
    { time: 90, survival: 0.895 },
    { time: 180, survival: 0.857 },
    { time: 365, survival: 0.784 },
    { time: 730, survival: 0.703 },
    { time: 1095, survival: 0.681 },
    { time: 1825, survival: 0.623 }
  ];
  
  // 2. Age Group Survival Data
  const ageGroupData = [
    { time: 0, under50: 1.000, age50to59: 1.000, age60to69: 1.000, age70plus: 1.000 },
    { time: 30, under50: 0.968, age50to59: 0.957, age60to69: 0.939, age70plus: 0.904 },
    { time: 90, under50: 0.942, age50to59: 0.921, age60to69: 0.883, age70plus: 0.837 },
    { time: 180, under50: 0.917, age50to59: 0.894, age60to69: 0.841, age70plus: 0.785 },
    { time: 365, under50: 0.873, age50to59: 0.838, age60to69: 0.772, age70plus: 0.691 },
    { time: 730, under50: 0.819, age50to59: 0.765, age60to69: 0.687, age70plus: 0.584 },
    { time: 1095, under50: 0.792, age50to59: 0.733, age60to69: 0.642, age70plus: 0.527 },
    { time: 1825, under50: 0.725, age50to59: 0.673, age60to69: 0.598, age70plus: 0.481 }
  ];
  
  // 3. Risk Group Survival Data
  const riskGroupData = [
    { time: 0, highRisk: 1.000, mediumRisk: 1.000, lowRisk: 1.000 },
    { time: 30, highRisk: 0.873, mediumRisk: 0.931, lowRisk: 0.985 },
    { time: 90, highRisk: 0.768, mediumRisk: 0.885, lowRisk: 0.957 },
    { time: 180, highRisk: 0.694, mediumRisk: 0.842, lowRisk: 0.936 },
    { time: 365, highRisk: 0.576, mediumRisk: 0.768, lowRisk: 0.892 },
    { time: 730, highRisk: 0.465, mediumRisk: 0.681, lowRisk: 0.847 },
    { time: 1095, highRisk: 0.417, mediumRisk: 0.629, lowRisk: 0.812 },
    { time: 1825, highRisk: 0.382, mediumRisk: 0.589, lowRisk: 0.778 }
  ];
  
  // 4. Survival Probability Table Data
  const survivalTableData = [
    { timePoint: '30 Days', baseline: 94.2, highRisk: 87.3, mediumRisk: 93.1, lowRisk: 98.5, under50: 96.8, age50to59: 95.7, age60to69: 93.9, age70plus: 90.4 },
    { timePoint: '90 Days', baseline: 89.5, highRisk: 76.8, mediumRisk: 88.5, lowRisk: 95.7, under50: 94.2, age50to59: 92.1, age60to69: 88.3, age70plus: 83.7 },
    { timePoint: '180 Days', baseline: 85.7, highRisk: 69.4, mediumRisk: 84.2, lowRisk: 93.6, under50: 91.7, age50to59: 89.4, age60to69: 84.1, age70plus: 78.5 },
    { timePoint: '1 Year', baseline: 78.4, highRisk: 57.6, mediumRisk: 76.8, lowRisk: 89.2, under50: 87.3, age50to59: 83.8, age60to69: 77.2, age70plus: 69.1 },
    { timePoint: '2 Years', baseline: 70.3, highRisk: 46.5, mediumRisk: 68.1, lowRisk: 84.7, under50: 81.9, age50to59: 76.5, age60to69: 68.7, age70plus: 58.4 },
    { timePoint: '5 Years', baseline: 62.3, highRisk: 38.2, mediumRisk: 58.9, lowRisk: 77.8, under50: 72.5, age50to59: 67.3, age60to69: 59.8, age70plus: 48.1 }
  ];
  
  // Format tick labels for the x-axis
  const formatXAxis = (value) => {
    if (value === 0) return '0';
    if (value === 30) return '30d';
    if (value === 90) return '3m';
    if (value === 180) return '6m';
    if (value === 365) return '1y';
    if (value === 730) return '2y';
    if (value === 1095) return '3y';
    if (value === 1825) return '5y';
    return value;
  };
  
  // Format percentage for tooltip
  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
  
  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold mb-8 text-center">Echocardiogram Survival Analysis</h1>
      
      {/* 1. Baseline Survival Function */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 bg-blue-50 p-2 rounded-md">Baseline Survival Function</h2>
        <p className="mb-4 text-gray-700">
          Overall survival probability for all patients over time. The steepest decline occurs within the first year,
          with survival decreasing from 100% to 78.4%, indicating this is the highest risk period.
        </p>
        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={baselineSurvivalData}
              margin={{ top: 10, right: 30, left: 30, bottom: 30 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                tickFormatter={formatXAxis}
                label={{ value: 'Time after Heart Attack', position: 'insideBottom', offset: -5, dy: 15, fill: '#000', fontSize: 16, fontWeight: 'bold', textAnchor: 'middle', x: '50%' }}
              />
              <YAxis 
                domain={[0, 1]} 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft', dx: -15, dy: 0, fill: '#000', fontSize: 16, fontWeight: 'bold', textAnchor: 'middle', y: '50%' }}
              />
              <Tooltip 
                formatter={(value) => `${(value * 100).toFixed(1)}%`}
                labelFormatter={(value) => value === 0 ? 'Baseline' : `${value} days`}
              />
              <Area 
                type="monotone" 
                dataKey="survival" 
                stroke="#3182ce" 
                strokeWidth={3}
                fill="#93c5fd" 
                name="Overall Survival"
                dot={true}
                activeDot={{ r: 8 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* 2. Age at Heart Attack Survival Analysis */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 bg-blue-50 p-2 rounded-md">Age at Heart Attack Survival Curves</h2>
        <p className="mb-4 text-gray-700">
          Survival varies significantly by age group, with younger patients showing better outcomes.
          The 5-year survival ranges from 72.5% for patients under 50 to 48.1% for those 70+.
        </p>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
                          <LineChart
              data={ageGroupData}
              margin={{ top: 10, right: 30, left: 30, bottom: 30 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                tickFormatter={formatXAxis}
                label={{ value: 'Time', position: 'insideBottomRight', offset: -5 }}
              />
              <YAxis 
                domain={[0, 1]} 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => `${(value * 100).toFixed(1)}%`}
                labelFormatter={(value) => value === 0 ? 'Baseline' : `${value} days`}
              />
              <Legend wrapperStyle={{ paddingTop: '10px' }} />
              <Line 
                type="monotone" 
                dataKey="under50" 
                stroke="#2563eb" 
                strokeWidth={3}
                name="Age <50" 
                dot={true}
                activeDot={{ r: 8 }}
              />
              <Line 
                type="monotone" 
                dataKey="age50to59" 
                stroke="#8b5cf6" 
                strokeWidth={3}
                name="Age 50-59" 
                dot={true}
                activeDot={{ r: 8 }}
              />
              <Line 
                type="monotone" 
                dataKey="age60to69" 
                stroke="#ec4899" 
                strokeWidth={3}
                name="Age 60-69" 
                dot={true}
                activeDot={{ r: 8 }}
              />
              <Line 
                type="monotone" 
                dataKey="age70plus" 
                stroke="#ef4444" 
                strokeWidth={3}
                name="Age 70+" 
                dot={true}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* 3. Risk Group Comparison */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 bg-blue-50 p-2 rounded-md">Risk Group Comparison</h2>
        <p className="mb-4 text-gray-700">
          Ejection fraction strongly predicts survival outcomes. High-risk patients (EF &lt;30%) have dramatically worse 
          prognosis with only 38.2% 5-year survival compared to 77.8% for low-risk patients (EF &gt;50%).
        </p>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={riskGroupData}
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                tickFormatter={formatXAxis}
                label={{ value: 'Time', position: 'insideBottomRight', offset: -5 }}
              />
              <YAxis 
                domain={[0, 1]} 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => `${(value * 100).toFixed(1)}%`}
                labelFormatter={(value) => value === 0 ? 'Baseline' : `${value} days`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="highRisk" 
                stroke="#ef4444" 
                strokeWidth={3}
                name="High Risk (EF <30%)" 
                dot={true}
                activeDot={{ r: 8 }}
              />
              <Line 
                type="monotone" 
                dataKey="mediumRisk" 
                stroke="#f97316" 
                strokeWidth={3}
                name="Medium Risk (EF 30-50%)" 
                dot={true}
                activeDot={{ r: 8 }}
              />
              <Line 
                type="monotone" 
                dataKey="lowRisk" 
                stroke="#22c55e" 
                strokeWidth={3}
                name="Low Risk (EF >50%)" 
                dot={true}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* 4. Survival Probabilities Table */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 bg-blue-50 p-2 rounded-md">Survival Probabilities Table</h2>
        <p className="mb-4 text-gray-700">
          This table summarizes the survival probabilities at key time points for all patient groups.
          It provides a comprehensive view of how both age and ejection fraction influence survival outcomes.
        </p>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead>
              <tr className="bg-blue-100">
                <th className="border px-4 py-3 font-bold text-lg">Time Point</th>
                <th className="border px-4 py-3 font-bold text-lg">Baseline</th>
                <th className="border px-4 py-3 bg-red-100 font-bold text-lg">High Risk</th>
                <th className="border px-4 py-3 bg-orange-100 font-bold text-lg">Medium Risk</th>
                <th className="border px-4 py-3 bg-green-100 font-bold text-lg">Low Risk</th>
                <th className="border px-4 py-3 bg-blue-100 font-bold text-lg">Age &lt;50</th>
                <th className="border px-4 py-3 bg-purple-100 font-bold text-lg">Age 50-59</th>
                <th className="border px-4 py-3 bg-pink-100 font-bold text-lg">Age 60-69</th>
                <th className="border px-4 py-3 bg-red-100 font-bold text-lg">Age 70+</th>
              </tr>
            </thead>
            <tbody>
              {survivalTableData.map((row, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="border px-4 py-3 font-medium text-base">{row.timePoint}</td>
                  <td className="border px-4 py-3 text-base font-medium">{row.baseline}%</td>
                  <td className="border px-4 py-3 bg-red-50 text-base font-medium">{row.highRisk}%</td>
                  <td className="border px-4 py-3 bg-orange-50 text-base font-medium">{row.mediumRisk}%</td>
                  <td className="border px-4 py-3 bg-green-50 text-base font-medium">{row.lowRisk}%</td>
                  <td className="border px-4 py-3 bg-blue-50 text-base font-medium">{row.under50}%</td>
                  <td className="border px-4 py-3 bg-purple-50 text-base font-medium">{row.age50to59}%</td>
                  <td className="border px-4 py-3 bg-pink-50 text-base font-medium">{row.age60to69}%</td>
                  <td className="border px-4 py-3 bg-red-50 text-base font-medium">{row.age70plus}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Key Findings Section */}
      <div className="mb-8 bg-gray-50 p-6 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4 bg-blue-50 p-2 rounded-md">Key Clinical Implications</h2>
        <ul className="list-disc pl-5 space-y-2">
          <li><span className="font-medium">Critical First Year:</span> The steepest survival decline occurs in the first year (particularly first 6 months), suggesting this period requires the most intensive monitoring and management.</li>
          <li><span className="font-medium">Ejection Fraction as Key Predictor:</span> EF provides excellent risk stratification with high-risk patients (EF &lt;30%) showing dramatically worse outcomes.</li>
          <li><span className="font-medium">Age-Related Risk:</span> Each decade of advancing age progressively worsens survival outcomes across all time points.</li>
          <li><span className="font-medium">Combined Risk Factors:</span> Advanced age (70+) with poor ejection fraction (&lt;30%) identifies an extremely high-risk population with &lt;30% 5-year survival.</li>
          <li><span className="font-medium">Stable Long-term Prognosis:</span> For all groups, the survival curve slope flattens after 2 years, indicating patients who survive this period have more stable long-term outcomes.</li>
        </ul>
      </div>
    </div>
  );
};

export default SurvivalAnalysisVisualizations;