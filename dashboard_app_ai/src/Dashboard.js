
import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const Dashboard = () => {
  const [data, setData] = useState({
    emotion_counts: {},
    cognitive_load_levels: {},
    face_presence_confidence: {},
    attention_time_level: {},
  });

  useEffect(() => {
    const fetchData = () => {
      fetch("/graph_update1.json")
        .then((res) => res.json())
        .then((json) => {
          if (json.status === "true") {
            setData(json);
          }
        })
        .catch((err) => {
          console.error("Error loading JSON:", err);
        });
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatChartData = (obj) =>
    Object.entries(obj).map(([key, value]) => ({
      name: key,
      value,
    }));

  const emotionData = formatChartData(data.emotion_counts);
  const cognitiveData = formatChartData(data.cognitive_load_levels);
  const faceConfidenceData = formatChartData(data.face_presence_confidence);
  const attentionTimeData = formatChartData(data.attention_time_level);

  const renderBarChart = (title, chartData, color) => (
    <div className="w-full max-w-[600px]">
      <h2 className="text-xl font-semibold items-center mb-4">{title}</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill={color} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="p-4">
      {/* Dashboard Title */}
      <h1 className="text-4xl font-bold items-center mb-8">
        AI Monitoring Dashboard
      </h1>

      {/* Grid with charts centered */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
        <div className="flex justify-center">{renderBarChart("Emotions", emotionData, "#8884d8")}</div>
        <div className="flex justify-center">{renderBarChart("Cognitive Load Levels", cognitiveData, "#82ca9d")}</div>
        <div className="flex justify-center">{renderBarChart("Face Presence Confidence", faceConfidenceData, "#ffc658")}</div>
        <div className="flex justify-center">{renderBarChart("Attention Levels", attentionTimeData, "#ff8042")}</div>
      </div>
    </div>
  );
};

export default Dashboard;

