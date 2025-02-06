// nextjs/pages/rag.tsx
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

export default function RagPage() {
  const router = useRouter();
  const { profileId } = router.query;
  
  const [topTen, setTopTen] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchTopTen = async () => {
    if (!profileId) return;
    setLoading(true);
    try {
      // Use your standard "flowProxy" or direct fetch 
      const res = await fetch(`/api/ragProxy?profile_id=${profileId}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to fetch top 10");
      }
      setTopTen(data.top_10 || []);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchTopTen();
  }, [profileId]);

  return (
    <div className="p-4 text-white bg-gray-900 min-h-screen">
      <h1 className="text-2xl font-bold">RAG Top 10 for Profile {profileId}</h1>
      {loading && <p>Loading...</p>}
      {error && <p className="text-red-400">{error}</p>}
      <div className="space-y-4 mt-4">
        {topTen.map((item, idx) => (
          <div key={idx} className="bg-gray-800 p-4 rounded">
            <p className="font-bold">Address: {item.address}</p>
            <p>Score: {item.score}</p>
            <p>Rationale: {item.rationale}</p>
          </div>
        ))}
      </div>
    </div>
  );
}