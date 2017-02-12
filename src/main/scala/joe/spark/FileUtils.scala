package joe.spark

object FileUtils {

  def resolvePathFromClasspath(path: String): String = {
    getClass.getResource(path).getPath
  }

}
