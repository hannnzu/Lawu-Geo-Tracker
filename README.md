# TEAM PROJECT DATA ENGINEERING KELOMPOK 8

***
**Open Data Source Analysis & Planning Project**

**Project Name**: Lawu Geo-Tracker: Integrasi Data Spasial & Elevasi Jalur Pendakian Gunung Lawu

**Created By**: Data Engineering Team 8

**Date**: February 19, 2026
**Version**: 1.0

***

**1. Executive Summary**

**1.1 Project Overview**


*   **Tujuan Project**: Membangun data geospasial untuk memetakan jalur pendakian Gunung Lawu beserta elevasi (ketinggian) dan estimasi waktu tempuh.
*   **Scope Project**: Integrasi data spasial rute dari OpenStreetMap, data topografi (elevasi), dan data aktivitas pendakian historis (GPX) pada 3 jalur utama (cemoro sewu, cemoro kandang, dan candi cetho)
*   **Expected Outcomes**: Dashboard analitik dan sistem rekomendasi rute
*   **Timeline**: 3 bulan (Februari - April 2026)

***

**2. Data Source Analysis**

**2.1 Data Pemerintah ([inarisk.bnpb.go.id](http://inarisk.bnpb.go.id/))**

**Source Details**
*   **Dataset Name**: Peta Indeks Rawan Bencana Area Gunung Lawu.
*   **URL/Access Point**: [inarisk.bnpb.go.id](http://inarisk.bnpb.go.id/)
*   **Data Owner**: Badan Nasional Penanggulangan Bencana
*   **Update Frequency**: Yearly

**Data Analysis**
*   **Format Data**: GeoJSON, WMS
*   **Volume Data**: ~15MB
*   **Time Coverage**: Indeks Risiko Bencana Terbaru (2024 - 2025)
*   **Data Quality**:
    * Completeness: 100% (Mencakup seluruh tutupan lahan di area Gunung Lawu).
    * Accuracy: Tinggi (Data resmi negara yang diverifikasi oleh para ahli kebencanaan dan GIS).
    * Consistency: Standar atribut spasial nasional.
    * Timeliness: Diperbarui secara berikala.

**2.2 Dataset wikiloc**

**Source Details**
*   **Dataset Name**: Jalur pendakian gunung lawu
*   **URL/Access Point**: [https://www.wikiloc.com/wikiloc/](https://www.wikiloc.com/wikiloc/map.do?sw=-89.999%2C-179.999&ne=89.999%2C179.999&q=Gunung%20Lawu%20Cemoro%20Sewu&fitMapToTrails=1&page=1)
*   **Data Owner**: Hiker comunity
*   **Update Frequency**: January 19, 2026

**Data Analysis**
*   **Format Data**: GPX
*   **Size & Dimensions**: ~2MB & ~4 coloumns dengan ~600 baris titik rekam.
*   **Data Fields**:
    * lat (latitude): Koordinat lintang.
    * lon (longitude): Koordinat bujur.
    * ele (elevation): Ketinggian dari permukaan laut (MDPL).
    * time (timestamp): Waktu dan tanggal titik WPS.
    * metadata: informasi dari pembuat track (seperti nama rute, author, ataupun link sumber)

*   **Quality Metrics**:
    * Missing Values: kemungkinan 0% karena semua data memiliki titik perekaman data lintang, bujur, serta ketinggian.
    * Data Types:
      * lat & lan: Float.
      * ele: Float.
      * time: Datetime.
      * metadata: String.
    * Consistency: Medium (data dibuat dari alat GPS ataupun device yang berbeda-beda)
    * Documentation Quality: Fair (GPX format standar berbasis XML yang bersifat self-describing).

**Public APIs**

**Source Details**
*   **API Name**: Overpass API (OpenStreetMap)
*   **Endpoint URL**: [overpass-api.de/api/interpreter](http://overpass-api.de/api/interpreter)
*   **Provider**: OpenStreetMap Foundation
*   **Authentication Method**:No Auth (Open Public API)

**API Analysis**
*   **Response Format**: JSON (OSM JSON) / GeoJSON
*   **Rate Limits**: ~10.000 requests / day
*   **Reliability**: 99.9% uptime
*   **Documentation Quality**:Very good.
*   **Cost**: Free

**2.4 Open Research Data**

**Source Details**
*   **Dataset Name**: ALOS World 3D - 30m (AW3D30) Digital Elevation Model.
*   **Repository**: OpenTopography
*   **Research Institution**: Japan Aerospace Exploration Agency (JAXA)
*   **Publication Date**: Version 3.2 (Januari 2021)

**Data Analysis**
*   **Format & Structure**: GeoTIFF
*   **Data Volume**: < 10 MB.
*   **Data Quality**:
    * Accuracy: very high (resolusi spasial 30 meter per piksel).
*   **Citation Requirements**: Japan Aerospace Exploration Agency (2021). ALOS World 3D 30 meter DEM. V3.2, Jan 2021. Distributed by OpenTopography.

**2.5 Data Cuaca**

**Source Details**
*   **API Name**: OpenWeatherMap Current Weather Data API
*   **Endpoint URL**: [api.openweathermap.org/data/2.5/weather](http://api.openweathermap.org/data/2.5/weather)
*   **Provider**: OpenWeather Ltd.
*   **Authentication Method**: API Key

**API Analysis**
*   **Response Format**: JSON
*   **Rate Limits**: 1.000 API calls/day
*   **Reliability**: 99.9% uptime
*   **Documentation Quality**: Medium-high.
*   **Cost**: Free.
