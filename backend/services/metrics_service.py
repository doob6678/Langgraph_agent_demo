import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
import threading

@dataclass
class RequestMetric:
    """请求指标"""
    timestamp: float
    response_time: float
    success: bool
    error_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """性能指标收集器"""
    
    def __init__(self):
        self.requests: List[RequestMetric] = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # 统计计数器
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        
        # 按端点统计
        self.endpoint_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": 0
        })
        
        # 按工具统计
        self.tool_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": 0
        })
    
    def record_request(self, endpoint: str = "unknown", metadata: Dict[str, Any] = None):
        """记录请求开始"""
        with self.lock:
            self.total_requests += 1
            self.endpoint_stats[endpoint]["count"] += 1
    
    def record_response(self, response_time: float, success: bool = True, 
                       endpoint: str = "unknown", error_type: str = "", 
                       metadata: Dict[str, Any] = None):
        """记录响应"""
        with self.lock:
            metric = RequestMetric(
                timestamp=time.time(),
                response_time=response_time,
                success=success,
                error_type=error_type,
                metadata=metadata or {}
            )
            self.requests.append(metric)
            
            # 更新统计
            self.total_response_time += response_time
            self.endpoint_stats[endpoint]["total_time"] += response_time
            
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                self.endpoint_stats[endpoint]["errors"] += 1
                
            # 保持最近1000条记录
            if len(self.requests) > 1000:
                self.requests = self.requests[-1000:]
    
    def record_error(self, error_type: str = "unknown", endpoint: str = "unknown"):
        """记录错误"""
        self.record_response(0.0, success=False, endpoint=endpoint, error_type=error_type)
    
    def record_tool_usage(self, tool_name: str, execution_time: float, success: bool = True):
        """记录工具使用情况"""
        with self.lock:
            self.tool_stats[tool_name]["count"] += 1
            self.tool_stats[tool_name]["total_time"] += execution_time
            if not success:
                self.tool_stats[tool_name]["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            # 基础指标
            metrics = {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests / max(self.total_requests, 1),
                "avg_response_time": self.total_response_time / max(self.successful_requests, 1),
                "requests_per_second": self.total_requests / max(uptime, 1)
            }
            
            # 端点统计
            endpoint_metrics = {}
            for endpoint, stats in self.endpoint_stats.items():
                endpoint_metrics[endpoint] = {
                    "count": stats["count"],
                    "avg_time": stats["total_time"] / max(stats["count"], 1),
                    "error_rate": stats["errors"] / max(stats["count"], 1)
                }
            metrics["endpoints"] = endpoint_metrics
            
            # 工具统计
            tool_metrics = {}
            for tool, stats in self.tool_stats.items():
                tool_metrics[tool] = {
                    "count": stats["count"],
                    "avg_time": stats["total_time"] / max(stats["count"], 1),
                    "error_rate": stats["errors"] / max(stats["count"], 1)
                }
            metrics["tools"] = tool_metrics
            
            # 最近请求统计
            if self.requests:
                recent_requests = self.requests[-100:]  # 最近100条
                recent_avg_time = sum(r.response_time for r in recent_requests) / len(recent_requests)
                recent_success_rate = sum(1 for r in recent_requests if r.success) / len(recent_requests)
                
                metrics["recent"] = {
                    "avg_response_time": recent_avg_time,
                    "success_rate": recent_success_rate,
                    "count": len(recent_requests)
                }
            
            return metrics
    
    def get_detailed_stats(self, limit: int = 100) -> Dict[str, Any]:
        """获取详细统计信息"""
        with self.lock:
            recent_requests = self.requests[-limit:]
            
            # 错误分析
            errors_by_type = defaultdict(int)
            errors_by_endpoint = defaultdict(int)
            
            for request in recent_requests:
                if not request.success and request.error_type:
                    errors_by_type[request.error_type] += 1
                    if "endpoint" in request.metadata:
                        errors_by_endpoint[request.metadata["endpoint"]] += 1
            
            # 响应时间分布
            response_times = [r.response_time for r in recent_requests if r.success]
            response_time_percentiles = {}
            if response_times:
                response_times.sort()
                n = len(response_times)
                response_time_percentiles = {
                    "p50": response_times[int(n * 0.5)],
                    "p90": response_times[int(n * 0.9)],
                    "p95": response_times[int(n * 0.95)],
                    "p99": response_times[int(n * 0.99)]
                }
            
            return {
                "recent_errors_by_type": dict(errors_by_type),
                "recent_errors_by_endpoint": dict(errors_by_endpoint),
                "response_time_percentiles": response_time_percentiles,
                "total_recent_requests": len(recent_requests)
            }
    
    def reset_metrics(self):
        """重置指标"""
        with self.lock:
            self.requests.clear()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_response_time = 0.0
            self.endpoint_stats.clear()
            self.tool_stats.clear()
            self.start_time = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        metrics = self.get_metrics()
        
        # 健康检查逻辑
        is_healthy = True
        issues = []
        
        if metrics["success_rate"] < 0.9:  # 成功率低于90%
            is_healthy = False
            issues.append(f"成功率过低: {metrics['success_rate']:.2%}")
        
        if metrics["avg_response_time"] > 5.0:  # 平均响应时间超过5秒
            is_healthy = False
            issues.append(f"响应时间过长: {metrics['avg_response_time']:.2f}s")
        
        if metrics["requests_per_second"] > 100:  # QPS过高
            is_healthy = False
            issues.append(f"QPS过高: {metrics['requests_per_second']:.2f}")
        
        return {
            "healthy": is_healthy,
            "issues": issues,
            "metrics_summary": {
                "success_rate": metrics["success_rate"],
                "avg_response_time": metrics["avg_response_time"],
                "total_requests": metrics["total_requests"]
            }
        }

# 创建全局实例
metrics_collector = MetricsCollector()