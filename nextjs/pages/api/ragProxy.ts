// nextjs/pages/api/ragProxy.ts
import type { NextApiRequest, NextApiResponse } from "next";

export default async function ragProxy(req: NextApiRequest, res: NextApiResponse) {
  const profileId = req.query.profile_id;
  if (!profileId) {
    return res.status(400).json({ detail: "Missing profile_id" });
  }

  try {
    const backendUrl = `http://localhost:8005/rag/top10?profile_id=${profileId}`;
    const backendRes = await fetch(backendUrl);
    const data = await backendRes.json();

    return res.status(backendRes.status).json(data);
  } catch (error) {
    console.error("ragProxy error:", error);
    return res.status(500).json({ detail: "Failed to fetch from backend" });
  }
}