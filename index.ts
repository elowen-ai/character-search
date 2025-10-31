import { z } from "zod";
import axios from "axios";
import { Router } from "express";

import models from "../../db";
import { Config } from "../../config";
import { serverOnly } from "../middleware";

const router = Router();

const MIN_CONFIDENCE = 41;

const getRawResult = async (q: string) => {
    // 3-255 characters, string
    const query = z.string().min(3).max(255).safeParse(q);

    if (!query.success) {
        return {
            records: [],
            query: { success: false, data: null },
            message: "Search query must be between 3 and 255 characters"
        };
    }

    const params = new URLSearchParams({ q: query.data, page: "1", page_size: "24", alpha: "0.7" });
    const py = await axios.get(
        `${Config.PythonBase}/character-search/paginated?${params.toString()}`,
    );

    const data = py.data as {
        results: Array<{ name: string; confidence?: number }>;
        pagination: any;
        query: string;
    };

    console.log(data.results);

    const filtered = data.results.filter(r => (r.confidence ?? 0) >= MIN_CONFIDENCE);
    const orderedNames = filtered.map(r => r.name);

    if (orderedNames.length === 0) return { records: [], query };

    const records = await models.instance.characters.findAsync(
        { name: { $in: orderedNames } },
        { allow_filtering: true },
    );

    return { records, query, orderedNames };
}

router.get("/character", async (req, res) => {
    try {
        const { records, query, message, orderedNames } = await getRawResult(req.query.q as string);

        if (!query.success) return res.status(400).json({ message });
        if (records.length === 0) return res.json({ results: [], query: query.data });

        const details = records.map((c: any) => ({
            id: c.id,
            wallet: c.wallet,
            name: c.name,
            description: c.description,
            tags: c.tags,
            avatar: c.avatar,
            likes: c.likes,
            dislikes: c.dislikes,
            premium: c.premium,
        }));

        const orderMap = new Map<string, number>();
        orderedNames!.forEach((n, i) => orderMap.set(n, i));
        details.sort((a: { name: string }, b: { name: string }) =>
            (orderMap.get(a.name) ?? 0) - (orderMap.get(b.name) ?? 0),
        );

        res.json({ results: details, query: query.data });
    } catch (error: any) {
        // TODO: Add logger for error handling
        console.error(error);
        res.status(error?.response?.status ?? 500).json({
            message: error?.response?.data?.message || error.message || "Internal server error",
        });
    }
});

router.get("/character/paginated", async (req, res) => {
    try {
        const query = z.string().min(3).max(255).safeParse(req.query.q);
        const page = z
            .string()
            .regex(/^\d+$/)
            .transform(v => parseInt(v, 10))
            .or(z.number())
            .default(1)
            .safeParse((req.query.page as string) ?? "1");
        const pageSize = z
            .string()
            .regex(/^\d+$/)
            .transform(v => parseInt(v, 10))
            .or(z.number())
            .default(24)
            .safeParse((req.query.page_size as string) ?? "24");

        if (!query.success) {
            res.status(400).json({
                message: "Search query must be between 3 and 255 characters",
            });
            return;
        }

        if (!page.success || page.data < 1) {
            res.status(400).json({ message: "Page number must be at least 1" });
            return;
        }

        if (!pageSize.success || pageSize.data < 1 || pageSize.data > 100) {
            res
                .status(400)
                .json({ message: "Page size must be between 1 and 100" });
            return;
        }

        const params = new URLSearchParams({
            q: query.data,
            page: String(page.data),
            page_size: String(pageSize.data),
            alpha: "0.7",
        });

        const response = await axios.get(
            `${Config.PythonBase}/character-search/paginated?${params.toString()}`,
        );

        const data = response.data as {
            results: Array<{ name: string; confidence?: number }>;
            pagination: any;
            query: string;
        };

        const filtered = data.results.filter(r => (r.confidence ?? 0) >= MIN_CONFIDENCE);
        const orderedNames = filtered.map(r => r.name);

        if (orderedNames.length === 0) {
            res.json({ results: [], pagination: data.pagination, query: data.query });
            return;
        }

        const characters = await models.instance.characters.findAsync(
            { name: { $in: orderedNames } },
            { allow_filtering: true },
        );

        // id, wallet, name, description, tags, avatar, likes, dislikes, premium
        const characterDetails = characters.map((c: any) => ({
            id: c.id,
            wallet: c.wallet,
            name: c.name,
            description: c.description,
            tags: c.tags,
            avatar: c.avatar,
            likes: c.likes,
            dislikes: c.dislikes,
            premium: c.premium,
        })) as {
            id: string;
            wallet: string;
            name: string;
            description: string;
            tags: string[];
            avatar: string;
            likes: number;
            dislikes: number;
            premium: boolean;
        }[];

        const orderMap = new Map<string, number>();
        orderedNames.forEach((n, i) => orderMap.set(n, i));
        characterDetails.sort((a, b) => (orderMap.get(a.name) ?? 0) - (orderMap.get(b.name) ?? 0));

        res.json({
            results: characterDetails,
            pagination: data.pagination,
            query: data.query,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Internal server error" });
    }
});

router.get("/character/full", serverOnly, async (req, res) => {
    try {
        const { records, query, message, orderedNames } = await getRawResult(req.query.q as string);

        if (!query.success) return res.status(400).json({ message });
        if (records.length === 0) return res.json({ results: [], query: query.data });

        const details = records.map((c: any) => ({
            id: c.id,
            wallet: c.wallet,
            name: c.name,
            description: c.description,
            personality: c.personality,
            speaking_style: c.speaking_style,
            samples: c.samples,
            tags: c.tags,
            avatar: c.avatar,
            likes: c.likes,
            dislikes: c.dislikes,
            premium: c.premium,
        }));

        const orderMap = new Map<string, number>();
        orderedNames!.forEach((n, i) => orderMap.set(n, i));
        details.sort((a: { name: string }, b: { name: string }) =>
            (orderMap.get(a.name) ?? 0) - (orderMap.get(b.name) ?? 0),
        );

        res.json({ results: details, query: query.data });
    } catch (error: any) {
        console.error(error);
        res.status(error?.response?.status ?? 500).json({
            message: error?.response?.data?.message || error.message || "Internal server error",
        });
    }
});

export default router;
