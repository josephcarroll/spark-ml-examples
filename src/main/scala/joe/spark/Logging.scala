package joe.spark

import org.apache.log4j.LogManager

trait Logging {

  @transient private val log = LogManager.getLogger(getClass.getSimpleName.stripSuffix("$"))
  @inline protected final def info(msg: => String) = log.info(msg)
  @inline protected final def debug(msg: => String) = log.debug(msg)
  @inline protected final def warn(msg: => String) = log.warn(msg)
  @inline protected final def error(msg: => String) = log.error(msg)
  @inline protected final def error(msg: => String, ex: => Throwable) = log.error(msg, ex)

}
